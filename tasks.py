"""
Task definitions and patient datasets for the Medical Triage environment.
Tasks: Easy (Task 1) → Medium (Task 2) → Hard (Task 3).
Each task has a deterministic grader returning score 0.0–1.0.
"""

from __future__ import annotations
import copy
from typing import Any, Dict, List, Tuple
from schemas import InternalPatient as Patient, Vitals


def _p(pid, name, age, gender, complaint, vitals_dict, history, meds, arrival, esi, routing):
    return Patient(
        patient_id=pid, name=name, age=age, gender=gender,
        chief_complaint=complaint,
        vitals=Vitals(**vitals_dict),
        medical_history=history,
        current_medications=meds,
        arrival_time_minutes=arrival,
        true_esi=esi,
        true_routing=routing,
    )


# ── ESI 1 — Immediate life threat ─────────────────────────────────────────
ESI1_PATIENTS = [
    _p("P001","James Thornton",58,"M","unresponsive, found at home by family",
       dict(bp_systolic=70,bp_diastolic=40,heart_rate=130,respiratory_rate=6,
            spo2=82.0,temperature=35.2,gcs=5,pain_scale=0),
       ["diabetes","hypertension"],["metformin","lisinopril"],0,1,"resuscitation_bay"),

    _p("P002","Maria Santos",34,"F","active seizure, not stopping",
       dict(bp_systolic=160,bp_diastolic=95,heart_rate=145,respiratory_rate=22,
            spo2=88.0,temperature=38.9,gcs=6,pain_scale=0),
       ["epilepsy"],["levetiracetam"],5,1,"resuscitation_bay"),

    _p("P003","David Kim",67,"M","sudden cardiac arrest, CPR in progress",
       dict(bp_systolic=0,bp_diastolic=0,heart_rate=0,respiratory_rate=0,
            spo2=70.0,temperature=36.5,gcs=3,pain_scale=0),
       ["CAD","prior MI"],["aspirin","atorvastatin"],2,1,"resuscitation_bay"),
]

# ── ESI 2 — High risk ─────────────────────────────────────────────────────
ESI2_PATIENTS = [
    _p("P004","Priya Patel",52,"F","worst headache of my life, sudden onset",
       dict(bp_systolic=190,bp_diastolic=110,heart_rate=98,respiratory_rate=18,
            spo2=97.0,temperature=37.1,gcs=14,pain_scale=9),
       ["hypertension"],["amlodipine"],10,2,"acute_care"),

    _p("P005","Robert Chen",61,"M","chest pressure radiating to left arm, sweating",
       dict(bp_systolic=145,bp_diastolic=88,heart_rate=105,respiratory_rate=20,
            spo2=94.0,temperature=37.0,gcs=15,pain_scale=8),
       ["hypertension","hyperlipidemia","smoker"],["aspirin"],3,2,"acute_care"),

    _p("P006","Aisha Mohammed",28,"F","severe shortness of breath, known asthma",
       dict(bp_systolic=130,bp_diastolic=85,heart_rate=118,respiratory_rate=28,
            spo2=91.0,temperature=37.5,gcs=15,pain_scale=7),
       ["asthma"],["salbutamol inhaler"],8,2,"acute_care"),

    _p("P007","Thomas Wright",44,"M","right-sided weakness and slurred speech, started 45 min ago",
       dict(bp_systolic=175,bp_diastolic=100,heart_rate=88,respiratory_rate=16,
            spo2=96.0,temperature=37.0,gcs=13,pain_scale=3),
       ["hypertension","atrial fibrillation"],["warfarin"],15,2,"acute_care"),
]

# ── ESI 3 — Urgent ────────────────────────────────────────────────────────
ESI3_PATIENTS = [
    _p("P008","Linda Park",39,"F","right lower quadrant pain for 12 hours, nausea",
       dict(bp_systolic=118,bp_diastolic=76,heart_rate=92,respiratory_rate=16,
            spo2=98.0,temperature=38.1,gcs=15,pain_scale=6),
       [],[],20,3,"fast_track"),

    _p("P009","Carlos Mendez",50,"M","blood in urine, back pain for 2 days",
       dict(bp_systolic=135,bp_diastolic=82,heart_rate=85,respiratory_rate=15,
            spo2=99.0,temperature=37.8,gcs=15,pain_scale=5),
       ["hypertension"],["hydrochlorothiazide"],25,3,"fast_track"),

    _p("P010","Emma Johnson",23,"F","severe vomiting for 8 hours, unable to keep fluids down",
       dict(bp_systolic=105,bp_diastolic=68,heart_rate=110,respiratory_rate=17,
            spo2=98.0,temperature=37.3,gcs=15,pain_scale=5),
       [],[],30,3,"fast_track"),
]

# ── ESI 4 — Less urgent ───────────────────────────────────────────────────
ESI4_PATIENTS = [
    _p("P011","George Baker",55,"M","mild ankle sprain after walking, can bear weight",
       dict(bp_systolic=122,bp_diastolic=78,heart_rate=72,respiratory_rate=14,
            spo2=99.0,temperature=36.8,gcs=15,pain_scale=3),
       ["type 2 diabetes"],["metformin"],35,4,"waiting_room"),

    _p("P012","Susan White",31,"F","sore throat and mild fever for 2 days",
       dict(bp_systolic=115,bp_diastolic=72,heart_rate=78,respiratory_rate=14,
            spo2=99.0,temperature=38.0,gcs=15,pain_scale=2),
       [],[],40,4,"waiting_room"),
]

# ── ESI 5 — Non-urgent ────────────────────────────────────────────────────
ESI5_PATIENTS = [
    _p("P013","Nancy Hill",27,"F","needs prescription refill, chronic condition stable",
       dict(bp_systolic=118,bp_diastolic=74,heart_rate=68,respiratory_rate=13,
            spo2=100.0,temperature=36.6,gcs=15,pain_scale=0),
       ["hypothyroidism"],["levothyroxine"],45,5,"waiting_room"),

    _p("P014","Brian Taylor",19,"M","minor cut on finger, controlled bleeding",
       dict(bp_systolic=120,bp_diastolic=76,heart_rate=70,respiratory_rate=14,
            spo2=100.0,temperature=36.7,gcs=15,pain_scale=1),
       [],[],50,5,"waiting_room"),
]

ALL_PATIENTS: Dict[str, Patient] = {
    p.patient_id: p for p in
    ESI1_PATIENTS + ESI2_PATIENTS + ESI3_PATIENTS + ESI4_PATIENTS + ESI5_PATIENTS
}

TASK_CONFIGS = {
    1: {
        "name": "Single Patient ESI Classification",
        "description": (
            "You are a triage nurse. Assess the presented patient and assign "
            "the correct Emergency Severity Index (ESI) level (1=most urgent, "
            "5=least urgent) and routing zone. You have one patient and 5 steps."
        ),
        "difficulty": "easy",
        "patient_ids": ["P005"],
        "max_steps": 5,
        "available_beds": {"resuscitation_bays": 2, "acute_care_beds": 8},
        "pass_threshold": 0.6,
    },
    2: {
        "name": "Multi-Patient Prioritization Queue",
        "description": (
            "Eight patients have arrived simultaneously. Triage all of them by "
            "assigning ESI levels and routing zones in correct priority order. "
            "Critical patients must be seen first. You have 20 steps."
        ),
        "difficulty": "medium",
        "patient_ids": ["P001","P004","P006","P008","P010","P011","P013","P007"],
        "max_steps": 20,
        "available_beds": {"resuscitation_bays": 2, "acute_care_beds": 6},
        "pass_threshold": 0.65,
    },
    3: {
        "name": "Full ED Surge Management",
        "description": (
            "A mass casualty event has flooded the ED. 15 patients need triage. "
            "Beds are extremely limited. Patients deteriorate if kept waiting. "
            "Assign ESI, route, request resources, and escalate life threats. "
            "You have 40 steps — every second counts."
        ),
        "difficulty": "hard",
        "patient_ids": [
            "P001","P002","P003","P004","P005",
            "P006","P007","P008","P009","P010",
            "P011","P012","P013","P014","P001",
        ],
        "max_steps": 40,
        "available_beds": {"resuscitation_bays": 1, "acute_care_beds": 3},
        "pass_threshold": 0.55,
    },
}


# ── Grading helpers ────────────────────────────────────────────────────────

def grade_esi_accuracy(assigned: int, true: int) -> Tuple[float, str]:
    diff = abs(assigned - true)
    if diff == 0:
        return 1.0, f"✅ Correct ESI {true}"
    elif diff == 1:
        return 0.4, f"⚠️ ESI off by 1 (assigned {assigned}, true {true})"
    elif diff == 2:
        return 0.1, f"❌ ESI off by 2 (assigned {assigned}, true {true})"
    else:
        return 0.0, f"🚨 Wrong ESI (assigned {assigned}, true {true})"


def grade_routing(assigned: str, true: str, esi: int) -> Tuple[float, str]:
    if assigned == true:
        return 1.0, f"✅ Correct routing: {true}"
    critical_zones = {"resuscitation_bay","acute_care"}
    noncritical_zones = {"fast_track","waiting_room"}
    if assigned in critical_zones and true in critical_zones:
        return 0.5, f"⚠️ Wrong critical zone (assigned {assigned}, true {true})"
    if assigned in noncritical_zones and true in noncritical_zones:
        return 0.5, f"⚠️ Wrong non-urgent zone"
    if true in critical_zones and assigned in noncritical_zones:
        return 0.0, f"🚨 Critical patient sent to {assigned}!"
    return 0.2, f"❌ Wrong routing ({assigned} vs {true})"


def is_critical_miss(assigned_esi: int, true_esi: int) -> bool:
    return true_esi <= 2 and assigned_esi >= (true_esi + 2)


def _normalize_requested_resources(requested: List[Any]) -> set[str]:
    """Normalize mixed resource payloads into a set of resource name strings."""
    normalized: set[str] = set()
    for item in requested or []:
        if isinstance(item, str):
            normalized.add(item)
            continue
        if isinstance(item, dict):
            # Accept common LLM payload shapes like {"resource": "ecg"}.
            for key in ("resource", "name", "type", "id"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    normalized.add(value)
                    break
    return normalized


def grade_resources(requested: List[str], true_esi: int) -> Tuple[float, str]:
    critical_resources = {"ecg","iv_access","blood_panel","ct_head","chest_xray"}
    requested_set = _normalize_requested_resources(requested)
    if true_esi == 1:
        needed = {"iv_access","blood_panel","ecg"}
        if needed.issubset(requested_set):
            return 1.0, "✅ Appropriate resources for ESI 1"
        overlap = len(needed & requested_set) / len(needed)
        return overlap * 0.8, "⚠️ Partial resources for critical patient"
    elif true_esi == 2:
        if len(requested_set & critical_resources) >= 2:
            return 0.8, "✅ Good resources for ESI 2"
        return 0.4, "⚠️ Insufficient resources for ESI 2"
    else:
        if len(requested_set & critical_resources) >= 3:
            return 0.5, "⚠️ Over-resourced low-acuity patient"
        return 1.0, "✅ Appropriate resource restraint"


def grade_task1(patients: Dict[str, Patient]) -> Dict:
    p = patients.get("P005")
    if p is None or p.assigned_esi is None:
        return {"final_score": 0.0, "breakdown": {}, "passed": False,
                "feedback": "No ESI assigned"}
    esi_score, _ = grade_esi_accuracy(p.assigned_esi, p.true_esi)
    routing_score, _ = grade_routing(
        p.assigned_routing or "", p.true_routing, p.true_esi)
    final = 0.6 * esi_score + 0.4 * routing_score
    return {
        "final_score": round(final, 3),
        "breakdown": {"esi_accuracy": esi_score, "routing_accuracy": routing_score},
        "passed": final >= TASK_CONFIGS[1]["pass_threshold"],
    }


def grade_task2(patients: Dict[str, Patient]) -> Dict:
    task_pids = TASK_CONFIGS[2]["patient_ids"]
    esi_scores, routing_scores, critical_misses = [], [], 0
    for pid in task_pids:
        p = patients.get(pid)
        if p is None or p.assigned_esi is None:
            esi_scores.append(0.0)
            routing_scores.append(0.0)
            if p and p.true_esi <= 2:
                critical_misses += 1
            continue
        es, _ = grade_esi_accuracy(p.assigned_esi, p.true_esi)
        rs, _ = grade_routing(p.assigned_routing or "", p.true_routing, p.true_esi)
        esi_scores.append(es)
        routing_scores.append(rs)
        if is_critical_miss(p.assigned_esi, p.true_esi):
            critical_misses += 1
    n = len(task_pids)
    mean_esi = sum(esi_scores) / n if n else 0
    mean_routing = sum(routing_scores) / n if n else 0
    coverage = sum(1 for pid in task_pids if patients.get(pid) and
                   patients[pid].assigned_esi) / n
    miss_penalty = critical_misses * 0.1
    final = max(0.0, 0.5*mean_esi + 0.3*mean_routing + 0.2*coverage - miss_penalty)
    return {
        "final_score": round(min(final, 1.0), 3),
        "breakdown": {
            "esi_accuracy": round(mean_esi, 3),
            "routing_accuracy": round(mean_routing, 3),
            "coverage": round(coverage, 3),
            "critical_miss_penalty": -round(miss_penalty, 3),
        },
        "critical_misses": critical_misses,
        "passed": final >= TASK_CONFIGS[2]["pass_threshold"],
    }


def grade_task3(patients: Dict[str, Patient], deteriorations: int, steps_used: int) -> Dict:
    esi_scores, routing_scores, resource_scores = [], [], []
    critical_misses = 0
    for p in patients.values():
        if p.assigned_esi is None:
            if p.true_esi <= 2:
                critical_misses += 1
            esi_scores.append(0.0)
            routing_scores.append(0.0)
            resource_scores.append(0.0)
            continue
        es, _ = grade_esi_accuracy(p.assigned_esi, p.true_esi)
        rs, _ = grade_routing(p.assigned_routing or "", p.true_routing, p.true_esi)
        resc, _ = grade_resources(p.resources_requested, p.true_esi)
        esi_scores.append(es)
        routing_scores.append(rs)
        resource_scores.append(resc)
        if is_critical_miss(p.assigned_esi, p.true_esi):
            critical_misses += 1
    n = len(patients)
    mean_esi = sum(esi_scores) / n if n else 0
    mean_routing = sum(routing_scores) / n if n else 0
    mean_resource = sum(resource_scores) / n if n else 0
    coverage = sum(1 for p in patients.values() if p.assigned_esi) / n if n else 0
    max_steps = TASK_CONFIGS[3]["max_steps"]
    efficiency = max(0.0, (max_steps - steps_used) / max_steps) * 0.1
    deterioration_penalty = deteriorations * 0.08
    miss_penalty = critical_misses * 0.12
    final = max(0.0, (
        0.35*mean_esi + 0.25*mean_routing + 0.2*mean_resource
        + 0.1*coverage + efficiency
        - deterioration_penalty - miss_penalty
    ))
    return {
        "final_score": round(min(final, 1.0), 3),
        "breakdown": {
            "esi_accuracy": round(mean_esi, 3),
            "routing_accuracy": round(mean_routing, 3),
            "resource_accuracy": round(mean_resource, 3),
            "coverage": round(coverage, 3),
            "efficiency_bonus": round(efficiency, 3),
            "deterioration_penalty": -round(deterioration_penalty, 3),
            "critical_miss_penalty": -round(miss_penalty, 3),
        },
        "critical_misses": critical_misses,
        "deteriorations": deteriorations,
        "passed": final >= TASK_CONFIGS[3]["pass_threshold"],
    }
