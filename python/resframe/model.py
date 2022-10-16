import resframe


class RCModel:
    def __init__(self) -> None:
        self.reservoire = resframe.Reservoire()

    def test_reservoire(self) -> None:
        test_run = self.reservoire.test_run()
        print(test_run)
