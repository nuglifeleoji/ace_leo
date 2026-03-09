#!/usr/bin/env python3
"""
Mastery-Based Curriculum Learning for FiNER — MACRO-CATEGORY version.

Strategy:
  1. Explode batched 4-question samples into single-question samples.
  2. Map each XBRL tag to one of ~12 macro categories (e.g., "Debt_Financing").
  3. Sort macro categories by total sample count (most common first).
  4. For each macro category, train on a shuffled mix of all its sub-labels
     until mastery (sliding window) is achieved at the CATEGORY level.
     → The model learns "how to distinguish tags WITHIN this category".
  5. After all categories done, evaluate on the test set.

Macro categories (12):
  1.  Debt_Financing           – DebtInstrument*, LineOfCredit*, LongTermDebt*, etc.
  2.  Equity_Shares            – CommonStock*, TreasuryStock*, StockRepurchase*, etc.
  3.  ShareBased_Compensation  – AllocatedShareBased*, ShareBasedCompensation*, etc.
  4.  Tax                      – EffectiveIncomeTaxRate*, IncomeTaxExpense*, etc.
  5.  Revenue_Contracts        – Revenues, ContractWithCustomer*, etc.
  6.  MA_BusinessCombinations  – BusinessCombination*, PaymentsToAcquire*, etc.
  7.  Leases_RealEstate        – OperatingLease*, LeaseAndRentalExpense, etc.
  8.  Intangibles_Goodwill_PPE – Goodwill*, Depreciation, FiniteLivedIntangible*, etc.
  9.  Restructuring_Impairment – RestructuringCharges, AssetImpairmentCharges, etc.
  10. Investments_Segments     – EquityMethod*, NumberOf*Segments, RelatedParty*, etc.
  11. Compensation_Benefits    – DefinedContributionPlan*, DefinedBenefitPlan*
  12. Contingencies_Legal_Other– LossContingency*, AccrualForEnvironmental*, etc.

Usage:
  cd /workspace/ace_leo
  source .env
  python -m eval.finance.run_mastery_macro \
      --save_path results/finer_mastery_macro \
      --api_provider together \
      --generator_model deepseek-ai/DeepSeek-V3.1 \
      --mastery_threshold 5 \
      --mastery_window 8 \
      --test_workers 20
"""

import os
import re
import json
import random
import argparse
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Dict, Any

from ace import ACE
from eval.finance.data_processor import DataProcessor, load_data
from utils import evaluate_test_set


# ─────────────────────────────────────────────────────────────
# Macro-category mapping
# Each key = XBRL tag, value = macro category name
# ─────────────────────────────────────────────────────────────

MACRO_CATEGORIES: Dict[str, str] = {
    # ── 1. Debt / Financing ─────────────────────────────────────────────────
    "DebtInstrumentInterestRateStatedPercentage":               "Debt_Financing",
    "DebtInstrumentFaceAmount":                                 "Debt_Financing",
    "LineOfCreditFacilityMaximumBorrowingCapacity":             "Debt_Financing",
    "DebtInstrumentBasisSpreadOnVariableRate1":                 "Debt_Financing",
    "DebtInstrumentCarryingAmount":                             "Debt_Financing",
    "LettersOfCreditOutstandingAmount":                         "Debt_Financing",
    "LineOfCredit":                                             "Debt_Financing",
    "LineOfCreditFacilityCurrentBorrowingCapacity":             "Debt_Financing",
    "LineOfCreditFacilityRemainingBorrowingCapacity":           "Debt_Financing",
    "DebtInstrumentRedemptionPricePercentage":                  "Debt_Financing",
    "LongTermDebtFairValue":                                    "Debt_Financing",
    "LongTermDebt":                                             "Debt_Financing",
    "DebtInstrumentUnamortizedDiscount":                        "Debt_Financing",
    "InterestExpenseDebt":                                      "Debt_Financing",
    "DebtInstrumentMaturityDate":                               "Debt_Financing",
    "DebtInstrumentConvertibleConversionPrice1":                "Debt_Financing",
    "LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage":"Debt_Financing",
    "RepaymentsOfDebt":                                         "Debt_Financing",
    "DebtInstrumentTerm":                                       "Debt_Financing",
    "DebtInstrumentInterestRateEffectivePercentage":            "Debt_Financing",
    "GainsLossesOnExtinguishmentOfDebt":                        "Debt_Financing",
    "DeferredFinanceCostsNet":                                  "Debt_Financing",
    "DebtWeightedAverageInterestRate":                          "Debt_Financing",
    "LineOfCreditFacilityInterestRateAtPeriodEnd":              "Debt_Financing",
    "DebtInstrumentFairValue":                                  "Debt_Financing",
    "DeferredFinanceCostsGross":                                "Debt_Financing",
    "AmortizationOfFinancingCosts":                             "Debt_Financing",
    "DerivativeFixedInterestRate":                              "Debt_Financing",
    "InterestExpense":                                          "Debt_Financing",
    "LineOfCreditFacilityCommitmentFeePercentage":              "Debt_Financing",

    # ── 2. Equity / Shares ──────────────────────────────────────────────────
    "AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount": "Equity_Shares",
    "StockIssuedDuringPeriodSharesNewIssues":                   "Equity_Shares",
    "CommonStockSharesAuthorized":                              "Equity_Shares",
    "PreferredStockSharesAuthorized":                           "Equity_Shares",
    "CommonStockParOrStatedValuePerShare":                      "Equity_Shares",
    "CommonStockDividendsPerShareDeclared":                     "Equity_Shares",
    "SaleOfStockNumberOfSharesIssuedInTransaction":             "Equity_Shares",
    "CommonStockCapitalSharesReservedForFutureIssuance":        "Equity_Shares",
    "TreasuryStockSharesAcquired":                              "Equity_Shares",
    "SaleOfStockPricePerShare":                                 "Equity_Shares",
    "TreasuryStockAcquiredAverageCostPerShare":                 "Equity_Shares",
    "TreasuryStockValueAcquiredCostMethod":                     "Equity_Shares",
    "ProceedsFromIssuanceOfCommonStock":                        "Equity_Shares",
    "StockRepurchasedDuringPeriodShares":                       "Equity_Shares",
    "StockRepurchasedAndRetiredDuringPeriodShares":             "Equity_Shares",
    "StockRepurchaseProgramAuthorizedAmount1":                  "Equity_Shares",
    "StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1":"Equity_Shares",
    "CommonStockSharesOutstanding":                             "Equity_Shares",
    "SharePrice":                                               "Equity_Shares",
    "PreferredStockDividendRatePercentage":                     "Equity_Shares",
    "ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1":    "Equity_Shares",

    # ── 3. Share-Based Compensation ─────────────────────────────────────────
    "AllocatedShareBasedCompensationExpense":                                                               "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod": "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1":                         "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized":                    "ShareBased_Compensation",
    "EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1": "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue": "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant":             "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue": "ShareBased_Compensation",
    "EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized":             "ShareBased_Compensation",
    "SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage":                "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross":                  "ShareBased_Compensation",
    "SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod":                            "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue": "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue": "ShareBased_Compensation",
    "EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense":                               "ShareBased_Compensation",
    "EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions": "ShareBased_Compensation",
    "ShareBasedCompensation":                                                                               "ShareBased_Compensation",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber": "ShareBased_Compensation",

    # ── 4. Tax ───────────────────────────────────────────────────────────────
    "EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate": "Tax",
    "EffectiveIncomeTaxRateContinuingOperations":                          "Tax",
    "IncomeTaxExpenseBenefit":                                             "Tax",
    "OperatingLossCarryforwards":                                          "Tax",
    "UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate":              "Tax",
    "UnrecognizedTaxBenefits":                                             "Tax",
    "CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption":          "Tax",

    # ── 5. Revenue / Contracts ───────────────────────────────────────────────
    "Revenues":                                              "Revenue_Contracts",
    "RevenueFromRelatedParties":                             "Revenue_Contracts",
    "RevenueFromContractWithCustomerExcludingAssessedTax":   "Revenue_Contracts",
    "ContractWithCustomerLiabilityRevenueRecognized":        "Revenue_Contracts",
    "ContractWithCustomerLiability":                         "Revenue_Contracts",
    "RevenueRemainingPerformanceObligation":                 "Revenue_Contracts",
    "CapitalizedContractCostAmortization":                   "Revenue_Contracts",

    # ── 6. M&A / Business Combinations ──────────────────────────────────────
    "PaymentsToAcquireBusinessesGross":                                                          "MA_BusinessCombinations",
    "BusinessCombinationConsiderationTransferred1":                                              "MA_BusinessCombinations",
    "BusinessAcquisitionPercentageOfVotingInterestsAcquired":                                   "MA_BusinessCombinations",
    "PaymentsToAcquireBusinessesNetOfCashAcquired":                                              "MA_BusinessCombinations",
    "BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued":                   "MA_BusinessCombinations",
    "DisposalGroupIncludingDiscontinuedOperationConsideration":                                  "MA_BusinessCombinations",
    "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles":  "MA_BusinessCombinations",
    "BusinessCombinationContingentConsiderationLiability":                                       "MA_BusinessCombinations",
    "BusinessCombinationAcquisitionRelatedCosts":                                                "MA_BusinessCombinations",
    "MinorityInterestOwnershipPercentageByParent":                                               "MA_BusinessCombinations",
    "MinorityInterestOwnershipPercentageByNoncontrollingOwners":                                 "MA_BusinessCombinations",
    "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill": "MA_BusinessCombinations",

    # ── 7. Leases / Real Estate ──────────────────────────────────────────────
    "OperatingLeasesRentExpenseNet":              "Leases_RealEstate",
    "LeaseAndRentalExpense":                      "Leases_RealEstate",
    "LesseeOperatingLeaseTermOfContract":         "Leases_RealEstate",
    "OperatingLeaseLiability":                    "Leases_RealEstate",
    "OperatingLeaseRightOfUseAsset":              "Leases_RealEstate",
    "OperatingLeaseWeightedAverageRemainingLeaseTerm1": "Leases_RealEstate",
    "AreaOfRealEstateProperty":                   "Leases_RealEstate",
    "NumberOfRealEstateProperties":               "Leases_RealEstate",

    # ── 8. Intangibles / Goodwill / PP&E ────────────────────────────────────
    "AmortizationOfIntangibleAssets":                      "Intangibles_Goodwill_PPE",
    "Goodwill":                                            "Intangibles_Goodwill_PPE",
    "FiniteLivedIntangibleAssetUsefulLife":                "Intangibles_Goodwill_PPE",
    "AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife": "Intangibles_Goodwill_PPE",
    "GoodwillImpairmentLoss":                              "Intangibles_Goodwill_PPE",
    "PropertyPlantAndEquipmentUsefulLife":                 "Intangibles_Goodwill_PPE",
    "Depreciation":                                        "Intangibles_Goodwill_PPE",

    # ── 9. Restructuring / Impairment ────────────────────────────────────────
    "RestructuringCharges":                   "Restructuring_Impairment",
    "RestructuringAndRelatedCostExpectedCost1": "Restructuring_Impairment",
    "AssetImpairmentCharges":                 "Restructuring_Impairment",

    # ── 10. Investments / Segments ───────────────────────────────────────────
    "ConcentrationRiskPercentage1":                            "Investments_Segments",
    "EquityMethodInvestmentOwnershipPercentage":               "Investments_Segments",
    "EquityMethodInvestments":                                 "Investments_Segments",
    "IncomeLossFromEquityMethodInvestments":                   "Investments_Segments",
    "NumberOfReportableSegments":                              "Investments_Segments",
    "NumberOfOperatingSegments":                               "Investments_Segments",
    "RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty": "Investments_Segments",
    "RelatedPartyTransactionAmountsOfTransaction":             "Investments_Segments",

    # ── 11. Compensation / Benefits (non-SBC) ────────────────────────────────
    "DefinedContributionPlanCostRecognized":   "Compensation_Benefits",
    "DefinedBenefitPlanContributionsByEmployer": "Compensation_Benefits",

    # ── 12. Contingencies / Legal / Other ────────────────────────────────────
    "SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense": "Contingencies_Legal_Other",
    "LossContingencyDamagesSoughtValue":        "Contingencies_Legal_Other",
    "LossContingencyEstimateOfPossibleLoss":    "Contingencies_Legal_Other",
    "LossContingencyAccrualAtCarryingValue":    "Contingencies_Legal_Other",
    "LossContingencyPendingClaimsNumber":       "Contingencies_Legal_Other",
    "AccrualForEnvironmentalLossContingencies": "Contingencies_Legal_Other",
    "GuaranteeObligationsMaximumExposure":      "Contingencies_Legal_Other",
    "PublicUtilitiesRequestedRateIncreaseDecreaseAmount": "Contingencies_Legal_Other",
    "CashAndCashEquivalentsFairValueDisclosure":"Contingencies_Legal_Other",
}

# Fallback: any unmapped label goes to a catch-all
FALLBACK_CATEGORY = "Other"


def get_macro_category(label: str) -> str:
    return MACRO_CATEGORIES.get(label, FALLBACK_CATEGORY)


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FiNER Mastery-Based Curriculum (Macro-Category level)"
    )
    p.add_argument("--save_path",         type=str, required=True)
    p.add_argument("--config_path",       type=str,
                   default="./eval/finance/data/sample_config.json")
    p.add_argument("--api_provider",      type=str, default="together",
                   choices=["sambanova", "together", "openai"])
    p.add_argument("--generator_model",   type=str, default="deepseek-ai/DeepSeek-V3.1")
    p.add_argument("--reflector_model",   type=str, default="deepseek-ai/DeepSeek-V3.1")
    p.add_argument("--curator_model",     type=str, default="deepseek-ai/DeepSeek-V3.1")
    p.add_argument("--max_tokens",        type=int, default=4096)
    p.add_argument("--mastery_threshold", type=int, default=5,
                   help="Correct answers needed in sliding window to declare mastery")
    p.add_argument("--mastery_window",    type=int, default=8,
                   help="Sliding window size for mastery check")
    p.add_argument("--max_num_rounds",    type=int, default=3)
    p.add_argument("--curator_frequency", type=int, default=1)
    p.add_argument("--playbook_token_budget", type=int, default=80000)
    p.add_argument("--test_workers",      type=int, default=20)
    p.add_argument("--seed",              type=int, default=42,
                   help="Random seed for shuffling samples within each macro category")
    p.add_argument("--initial_playbook_path", type=str, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Parsing: explode 4-question batches → single-question samples
# (identical to run_mastery.py)
# ─────────────────────────────────────────────────────────────

def parse_batch_to_singles(raw_sample: Dict) -> List[Dict]:
    """
    Given a raw FiNER sample with 4 batched questions, return up to 4
    single-question dicts:  {context, question, target, label, macro_category}
    """
    context    = raw_sample.get("context", "")
    target_str = raw_sample.get("target", "")
    targets    = [t.strip() for t in target_str.split(",")]

    q_positions = list(re.finditer(r'\n(\d+)\.\s+What is best tag', context))
    if not q_positions:
        return []

    header = context[: q_positions[0].start()].strip()
    header_single = re.sub(
        r'Answer the following \d+ independent questions by providing only\s+'
        r'\d+ US GAAP tags answers in the order of the questions\.'
        r'.*?Provide nothing else\.',
        'Answer the following question by providing only 1 US GAAP tag. '
        'Provide nothing else.',
        header,
        flags=re.DOTALL,
    )

    singles = []
    for i, match in enumerate(q_positions):
        if i >= len(targets):
            break

        q_start = match.start() + 1
        if i + 1 < len(q_positions):
            q_end = q_positions[i + 1].start()
        else:
            tail_match = re.search(r'\nOutput US GAAP tags:', context[match.start():])
            q_end = (match.start() + tail_match.start()) if tail_match else len(context)

        q_text = context[q_start:q_end].strip()
        target = targets[i]

        single_prompt = (
            f"{header_single}\n"
            f"{q_text}\n"
            f"Output US GAAP tag:"
        )

        singles.append({
            "context":        "",
            "question":       single_prompt,
            "target":         target,
            "label":          target,
            "macro_category": get_macro_category(target),
        })

    return singles


# ─────────────────────────────────────────────────────────────
# Mastery curriculum (macro-category level)
# ─────────────────────────────────────────────────────────────

def run_mastery_curriculum_macro(
    ace_system:        ACE,
    all_singles:       List[Dict],
    data_processor,
    config_params:     Dict[str, Any],
    save_path:         str,
    usage_log_path:    str,
    log_dir:           str,
    mastery_threshold: int,
    mastery_window:    int,
    seed:              int = 42,
) -> Dict[str, Any]:
    """
    Iterate over macro categories (sorted by total sample count, desc).
    Within each category, samples are shuffled so the model sees a MIXED
    stream of different fine-grained tags — forcing it to learn intra-category
    distinctions.  Mastery is checked at the category level.
    """
    # ── Group samples by macro category ─────────────────────
    cat_to_samples: Dict[str, List[Dict]] = defaultdict(list)
    for s in all_singles:
        cat_to_samples[s["macro_category"]].append(s)

    sorted_cats = sorted(cat_to_samples, key=lambda c: -len(cat_to_samples[c]))

    total_cats  = len(sorted_cats)
    global_step = 0
    cat_stats   = {}

    print(f"\n{'='*65}")
    print(f"MACRO-CATEGORY MASTERY CURRICULUM  ({total_cats} categories)")
    print(f"Window={mastery_window}, Threshold={mastery_threshold}")
    print(f"{'='*65}\n")

    playbook_dir = os.path.join(save_path, "intermediate_playbooks")
    os.makedirs(playbook_dir, exist_ok=True)

    rng = random.Random(seed)

    for cat_idx, cat in enumerate(sorted_cats):
        samples     = list(cat_to_samples[cat])   # copy
        rng.shuffle(samples)                       # shuffle for mixed-tag stream
        n_available = len(samples)

        # Count unique fine-grained tags in this category
        subtags = sorted({s["label"] for s in samples})
        threshold = min(mastery_threshold, max(2, n_available // 4))
        window    = deque(maxlen=mastery_window)
        n_used    = 0
        mastered  = False

        print(f"\n[{cat_idx+1}/{total_cats}]  Category: {cat}")
        print(f"  Sub-tags ({len(subtags)}): {', '.join(subtags[:6])}"
              + (" …" if len(subtags) > 6 else ""))
        print(f"  Total samples: {n_available} | threshold: {threshold}")

        for sample in samples:
            global_step += 1
            step_id = f"macro_{cat_idx}_s{n_used}"

            _, _, tracking = ace_system._train_single_sample(
                task_dict      = sample,
                data_processor = data_processor,
                step_id        = step_id,
                epoch          = 1,
                step           = global_step,
                usage_log_path = usage_log_path,
                log_dir        = log_dir,
                config_params  = config_params,
                total_samples  = len(all_singles),
            )

            pre_correct = tracking["pre_train_result"]["is_correct"]
            window.append(int(pre_correct))
            n_used += 1

            window_str = "".join("✓" if x else "✗" for x in window)
            print(f"  step {n_used:3d}/{n_available} | "
                  f"tag={sample['label'][:40]:<40} | "
                  f"correct={pre_correct} | "
                  f"[{window_str}] {sum(window)}/{len(window)}")

            if (len(window) >= min(mastery_window, threshold + 1)
                    and sum(window) >= threshold):
                mastered = True
                print(f"  ✅  MASTERED  {cat}  after {n_used} samples!")
                break

        if not mastered:
            print(f"  ⚠️  Exhausted all {n_available} samples for {cat} "
                  f"(best window: {sum(window)}/{len(window)})")

        cat_stats[cat] = {
            "n_used":               n_used,
            "n_available":          n_available,
            "threshold":            threshold,
            "mastered":             mastered,
            "final_window_correct": int(sum(window)),
            "n_subtags":            len(subtags),
            "subtags":              subtags,
        }

        # Save intermediate playbook after each category
        pb_path = os.path.join(
            playbook_dir, f"playbook_after_cat_{cat_idx:02d}_{cat}.txt"
        )
        with open(pb_path, "w") as f:
            f.write(ace_system.playbook)

    return cat_stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder  = f"ace_run_{timestamp}_finer_mastery_macro"
    save_path   = os.path.join(args.save_path, run_folder)
    log_dir     = os.path.join(save_path, "detailed_llm_logs")
    playbook_dir = os.path.join(save_path, "intermediate_playbooks")
    os.makedirs(log_dir,      exist_ok=True)
    os.makedirs(playbook_dir, exist_ok=True)
    usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")

    print(f"\n{'='*65}")
    print(f"FiNER Mastery-Based Curriculum  (MACRO-CATEGORY version)")
    print(f"Save path : {save_path}")
    print(f"Model     : {args.generator_model}")
    print(f"Mastery   : {args.mastery_threshold}/{args.mastery_window}")
    print(f"Seed      : {args.seed}")
    print(f"{'='*65}\n")

    # ── Load data ────────────────────────────────────────────
    with open(args.config_path) as f:
        task_config = json.load(f)

    finer_cfg    = task_config["finer"]
    train_raw    = load_data(finer_cfg["train_data"])
    test_raw     = load_data(finer_cfg["test_data"])

    data_processor = DataProcessor(task_name="finer")
    test_samples   = data_processor.process_task_data(test_raw)

    # ── Explode train batches → single-question samples ─────
    all_singles: List[Dict] = []
    for raw in train_raw:
        all_singles.extend(parse_batch_to_singles(raw))

    print(f"Train batches : {len(train_raw)}")
    print(f"Single Qs     : {len(all_singles)}")

    # Category summary
    cat_counts: Dict[str, int] = defaultdict(int)
    subtag_counts: Dict[str, set] = defaultdict(set)
    for s in all_singles:
        cat_counts[s["macro_category"]] += 1
        subtag_counts[s["macro_category"]].add(s["label"])

    # Warn about unmapped labels
    unmapped = [s["label"] for s in all_singles
                if s["macro_category"] == FALLBACK_CATEGORY]
    if unmapped:
        print(f"\n⚠  {len(set(unmapped))} unmapped labels → '{FALLBACK_CATEGORY}':")
        for lbl in sorted(set(unmapped)):
            print(f"   {lbl}")

    print(f"\nMacro categories ({len(cat_counts)}):")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30}  {cnt:4d} samples, "
              f"{len(subtag_counts[cat]):3d} sub-tags")

    # ── Load initial playbook (optional) ────────────────────
    initial_playbook = None
    if args.initial_playbook_path and os.path.exists(args.initial_playbook_path):
        with open(args.initial_playbook_path) as f:
            initial_playbook = f.read()
        print(f"\nInitial playbook: {args.initial_playbook_path}")
    else:
        print("\nInitial playbook: empty")

    # ── ACE system ───────────────────────────────────────────
    ace_system = ACE(
        api_provider      = args.api_provider,
        generator_model   = args.generator_model,
        reflector_model   = args.reflector_model,
        curator_model     = args.curator_model,
        max_tokens        = args.max_tokens,
        initial_playbook  = initial_playbook,
    )

    config_params = {
        "max_num_rounds":    args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "token_budget":      args.playbook_token_budget,
        "task_name":         "finer_mastery_macro",
        "use_json_mode":     False,
        "no_ground_truth":   False,
        "save_dir":          args.save_path,
        "test_workers":      args.test_workers,
        "eval_steps":        9999,
        "save_steps":        9999,
        "use_bulletpoint_analyzer": False,
        "bulletpoint_analyzer_threshold": 0.90,
    }

    # Save run config
    with open(os.path.join(save_path, "run_config.json"), "w") as f:
        json.dump({
            "mastery_threshold": args.mastery_threshold,
            "mastery_window":    args.mastery_window,
            "seed":              args.seed,
            "generator_model":   args.generator_model,
            "api_provider":      args.api_provider,
            "train_batches":     len(train_raw),
            "single_questions":  len(all_singles),
            "n_macro_categories": len(cat_counts),
        }, f, indent=2)

    # ── Baseline test (empty playbook) ───────────────────────
    print(f"\n{'='*65}")
    print("BASELINE TEST (empty playbook)")
    print(f"{'='*65}")
    baseline_results, _ = evaluate_test_set(
        data_processor = data_processor,
        generator      = ace_system.generator,
        playbook       = ace_system.playbook,
        test_samples   = test_samples,
        max_tokens     = args.max_tokens,
        log_dir        = log_dir,
        max_workers    = args.test_workers,
    )
    baseline_acc = baseline_results["accuracy"]
    print(f"Baseline Test Accuracy: {baseline_acc:.4f}")

    # ── Macro mastery curriculum ─────────────────────────────
    cat_stats = run_mastery_curriculum_macro(
        ace_system        = ace_system,
        all_singles       = all_singles,
        data_processor    = data_processor,
        config_params     = config_params,
        save_path         = save_path,
        usage_log_path    = usage_log_path,
        log_dir           = log_dir,
        mastery_threshold = args.mastery_threshold,
        mastery_window    = args.mastery_window,
        seed              = args.seed,
    )

    # ── Save final playbook ──────────────────────────────────
    final_pb_path = os.path.join(save_path, "final_playbook.txt")
    with open(final_pb_path, "w") as f:
        f.write(ace_system.playbook)
    print(f"\nFinal playbook saved: {final_pb_path}")

    # ── Summary ──────────────────────────────────────────────
    total_used     = sum(v["n_used"]  for v in cat_stats.values())
    total_mastered = sum(1 for v in cat_stats.values() if v["mastered"])
    print(f"\n{'='*65}")
    print(f"MASTERY SUMMARY")
    print(f"{'='*65}")
    print(f"Categories mastered : {total_mastered} / {len(cat_stats)}")
    print(f"Total samples used  : {total_used} / {len(all_singles)}")
    print(f"\nPer-category breakdown:")
    hdr = f"{'Category':<30} {'avail':>6} {'used':>5} {'thr':>4} {'subtags':>7} {'mastered':>8}"
    print(hdr)
    print("-" * len(hdr))
    for cat, stat in sorted(cat_stats.items(), key=lambda x: -x[1]["n_used"]):
        print(f"{cat:<30} {stat['n_available']:>6} {stat['n_used']:>5} "
              f"{stat['threshold']:>4} {stat['n_subtags']:>7} "
              f"{'✅' if stat['mastered'] else '❌':>8}")

    # Save category stats JSON
    stats_path = os.path.join(save_path, "category_mastery_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "cat_stats":       cat_stats,
            "total_used":      total_used,
            "total_mastered":  total_mastered,
            "total_categories": len(cat_stats),
            "baseline_acc":    baseline_acc,
        }, f, indent=2)

    # ── Final test eval ──────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL TEST (macro mastery playbook)")
    print(f"{'='*65}")
    final_results, _ = evaluate_test_set(
        data_processor = data_processor,
        generator      = ace_system.generator,
        playbook       = ace_system.playbook,
        test_samples   = test_samples,
        max_tokens     = args.max_tokens,
        log_dir        = log_dir,
        max_workers    = args.test_workers,
    )
    final_acc = final_results["accuracy"]
    print(f"Baseline Test Accuracy : {baseline_acc:.4f}")
    print(f"Final    Test Accuracy : {final_acc:.4f}")
    print(f"Delta                  : {final_acc - baseline_acc:+.4f}")

    # Update stats file
    with open(stats_path) as f:
        stats = json.load(f)
    stats["final_acc"] = final_acc
    stats["delta"]     = final_acc - baseline_acc
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*65}")
    print(f"All results saved to: {save_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
