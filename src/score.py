import os

class ScoreManager:
    def __init__(self, filename="highscore.txt"):
        self.filename = filename
        self.high_score = self._load_high_score()
        self.current_score = None

    def _load_high_score(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    return float(f.read().strip())
            except:
                return 0.0
        return 0.0

    def _save_high_score(self):
        with open(self.filename, "w") as f:
            f.write(str(self.high_score))

    def update(self, score):
        """
        Update current score and high score if beaten.
        """
        self.current_score = score

        if score > self.high_score:
            self.high_score = score
            self._save_high_score()
            return True  

        return False

    def reset_session(self):
        self.current_score = None

    def progress_ratio(self):
        """
        Returns value between 0 and 1 for progress bar.
        """
        if self.high_score <= 0 or self.current_score is None:
            return 0.0
        return min(self.current_score / self.high_score, 1.0)
