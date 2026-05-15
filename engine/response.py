"""
Phase 4 — Response Engine
Maps detected fault types to specific corrective grid actions.
Uses confidence score to decide automated action vs human alert.
"""

FAULT_LABELS = {
    0: "Normal",
    1: "Overvoltage",
    2: "Voltage Sag",
    3: "Overload",
    4: "Spike",
    5: "Line Fault"
}

FAULT_ACTIONS = {
    1: "Trigger relay trip on feeder line",
    2: "Switch to backup supply / raise transformer tap",
    3: "Shed non-critical load (load shedding)",
    4: "Insert surge arrestor, log event for maintenance",
    5: "Isolate faulty segment, reroute power via backup path"
}


def get_response(fault_type: int, confidence: float) -> str:
    if confidence < 0.6:
        possible = FAULT_LABELS.get(fault_type, "Unknown")
        return f"LOW CONFIDENCE ({confidence:.0%}) — Human review required. Possible fault: {possible}"
    label = FAULT_LABELS.get(fault_type, "Unknown")
    action = FAULT_ACTIONS.get(fault_type, "Investigate manually")
    return f"{label} detected → {action}"


def get_severity(fault_type: int) -> str:
    severity_map = {1: "HIGH", 2: "MEDIUM", 3: "HIGH", 4: "MEDIUM", 5: "CRITICAL"}
    return severity_map.get(fault_type, "LOW")
