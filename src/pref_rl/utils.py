class LinearSchedule:
    def __init__(self, start: float, end: float = 0):
        self.start = start
        self.end = end


    def __call__(self, progress_remaining: float):
        return self.end + progress_remaining * (self.start - self.end)
