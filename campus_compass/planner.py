"""Study-plan generation and user-directed plan changes."""

from __future__ import annotations

from datetime import date, timedelta


def build_study_plan(
    subjects: list[dict[str, str | int]],
    hours_per_day: float,
    start_date: date | None = None,
) -> list[dict[str, str | int]]:
    today = start_date or date.today()
    valid_subjects = []
    for subject in subjects:
        try:
            exam_date = date.fromisoformat(str(subject["exam_date"]))
        except (KeyError, ValueError):
            continue
        if exam_date >= today:
            valid_subjects.append({**subject, "parsed_exam_date": exam_date})
    if not valid_subjects or hours_per_day <= 0:
        return []

    latest_exam = max(item["parsed_exam_date"] for item in valid_subjects)
    daily_minutes = max(30, int(hours_per_day * 60))
    session_minutes = min(50, daily_minutes)
    plan: list[dict[str, str | int]] = []
    day = today
    while day <= latest_exam:
        eligible = [item for item in valid_subjects if item["parsed_exam_date"] >= day]
        eligible.sort(key=lambda item: (item["parsed_exam_date"], -int(item.get("priority", 3))))
        minutes_left = daily_minutes
        for item in eligible:
            if minutes_left < 25:
                break
            duration = min(session_minutes, minutes_left)
            subject_name = str(item["name"])
            exam_in = (item["parsed_exam_date"] - day).days
            if exam_in == 0:
                focus = "Final review and confidence check"
            elif exam_in <= 2:
                focus = "Practice questions and weak areas"
            elif day.weekday() == 6:
                focus = "Weekly recap and spaced retrieval"
            else:
                focus = "Core concepts, active recall, and practice"
            plan.append(
                {
                    "date": day.isoformat(),
                    "subject": subject_name,
                    "duration_min": duration,
                    "focus": focus,
                    "status": "Planned",
                }
            )
            minutes_left -= duration
        day += timedelta(days=1)
    return plan


def apply_plan_change(
    plan: list[dict[str, str | int]],
    action: str,
    target_index: int | None = None,
    replacement: dict[str, str | int] | None = None,
) -> list[dict[str, str | int]]:
    updated = [dict(session) for session in plan]
    if action == "add" and replacement:
        updated.append(replacement)
    elif action == "remove" and target_index is not None and 0 <= target_index < len(updated):
        updated.pop(target_index)
    elif action == "replace" and target_index is not None and replacement:
        if 0 <= target_index < len(updated):
            updated[target_index] = replacement
    return sorted(updated, key=lambda item: (str(item.get("date", "")), str(item.get("subject", ""))))