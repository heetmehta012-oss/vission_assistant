"""
core/scene_builder.py
=====================
Converts a list of detected objects into natural-language scene descriptions.
Groups objects by type and position for clear, concise narration.
"""

from collections import defaultdict


# Human-friendly plural forms
PLURALS = {
    "person": "people",
    "chair": "chairs",
    "table": "tables",
    "laptop": "laptops",
    "cell phone": "phones",
    "cup": "cups",
    "bottle": "bottles",
    "book": "books",
    "car": "cars",
    "dog": "dogs",
    "cat": "cats",
}

def pluralize(word, count):
    if count == 1:
        return word
    return PLURALS.get(word, word + "s")


# Priority order for narration (most important first)
PRIORITY_ORDER = [
    "person", "dog", "cat", "car", "bicycle",
    "traffic light", "stop sign", "stairs",
    "chair", "table", "door", "couch", "bed",
    "laptop", "monitor", "tv", "keyboard", "mouse",
    "cell phone", "book", "cup", "bottle", "backpack",
]


class SceneBuilder:
    def build_description(self, detections):
        """
        Takes a list of detection dicts and returns a spoken description string.
        Example: "Two people ahead. Chair on the left. Laptop on the right."
        """
        if not detections:
            return "I don't see any recognizable objects right now."

        # Group by label + position
        grouped = defaultdict(lambda: defaultdict(int))
        for d in detections:
            grouped[d["label"]][d["position"]] += 1

        # Sort by priority
        sorted_labels = sorted(
            grouped.keys(),
            key=lambda x: PRIORITY_ORDER.index(x) if x in PRIORITY_ORDER else 999,
        )

        phrases = []
        for label in sorted_labels:
            positions = grouped[label]
            for position, count in positions.items():
                obj_str = pluralize(label, count)
                count_str = self._number_word(count)

                if position == "ahead":
                    phrase = f"{count_str} {obj_str} ahead"
                elif position == "left":
                    phrase = f"{count_str} {obj_str} on the left"
                elif position == "right":
                    phrase = f"{count_str} {obj_str} on the right"
                else:
                    phrase = f"{count_str} {obj_str} nearby"

                phrases.append(phrase.capitalize())

        # Join into a spoken sentence
        description = ". ".join(phrases) + "."
        return description

    def build_person_description(self, detections):
        """Focused description for 'who is there' command."""
        people = [d for d in detections if d["label"] == "person"]
        if not people:
            return "I don't see anyone in front of you."
        count = len(people)
        positions = [p["position"] for p in people]
        pos_str = ", ".join(set(positions))
        noun = pluralize("person", count)
        return f"I can see {self._number_word(count)} {noun} — {pos_str}."

    def _number_word(self, n):
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
        return words.get(n, str(n))
