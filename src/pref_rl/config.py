class ConstantSchedule:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, progress_remaining: float, **kwargs):
        return self.value


class LinearSchedule:
    def __init__(self, start: float, end: float = 0):
        self.start = start
        self.end = end

    def __call__(self, progress_remaining: float, **kwargs):
        return self.end + progress_remaining * (self.start - self.end)


class ExponentialSchedule:
    def __init__(self, start: float, end: float = 0, decay: float = 0.5):
        self.start = start
        self.end = end
        self.decay = decay

    def __call__(self, progress_remaining: float, **kwargs):
        return self.end + (self.start - self.end) * (self.decay ** (1 - progress_remaining))


class PiecewiseConstantSchedule:
    def __init__(self, pieces: list[tuple[int, int]]):
        self.pieces = sorted(pieces, key=lambda p: p[0], reverse=True)

    def __call__(self, progress_remaining: float, num_timesteps: int, **kwargs):
        for step, value in self.pieces:
            if num_timesteps >= step:
                return value
