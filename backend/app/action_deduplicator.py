from models import Action


class ActionDeduplicator:

    def deduplicate(self, actions):

        unique = []
        seen = set()

        for action in actions:

            key = (
                action.type,
                action.text.lower().strip()
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(action)

        return unique