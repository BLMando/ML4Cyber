from enum import Enum


def _to_inch(n, dpi=300) -> float:
    return round(n / dpi, 1)


def PixToInch(width, height, dpi):
    return (_to_inch(width, dpi), _to_inch(height, dpi))


class FigSize(Enum):
    _dpi = 300
    DPI = _dpi
    XXS1_1 = PixToInch(500, 500, _dpi)
    XS1_1 = PixToInch(1000, 1000, _dpi)
    S1_1 = PixToInch(1500, 1500, _dpi)
    M1_1 = PixToInch(2000, 2000, _dpi)
    L1_1 = PixToInch(2500, 2500, _dpi)
    XL1_1 = PixToInch(3000, 3000, _dpi)
    XXL1_1 = PixToInch(5000, 5000, _dpi)
    XXXL1_1 = PixToInch(10000, 10000, _dpi)
    ENORMOUS1_1 = PixToInch(15000, 15000, _dpi)
    XE1_1 = PixToInch(50000, 50000, _dpi)
    XXS16_9 = (1.7, 0.9)  #  500 × 281
    XS16_9 = (3.3, 1.9)  # 1000 × 562
    S16_9 = (5.0, 2.8)  # 1500 × 844
    M16_9 = (6.7, 3.4)  # 2000 × 1125
    L16_9 = (8.3, 4.2)  # 2500 × 1406
    XL16_9 = (10.0, 5.0)  # 3000 × 1688
    XXL16_9 = (11.7, 5.9)  # 3500 × 1969
    XXXL16_9 = (13.3, 6.8)  # 4000 × 2250
    ENORMOUS16_9 = (16.7, 9.4)  # 5000 × 2812
    XE16_9 = (33.3, 18.8)  # 10000 × 5625
    XXS4_3 = (1.7, 1.3)  #  500 × 375
    XS4_3 = (3.3, 2.5)  # 1000 × 750
    S4_3 = (5.0, 3.8)  # 1500 × 1125
    M4_3 = (6.7, 5.0)  # 2000 × 1500
    L4_3 = (8.3, 6.3)  # 2500 × 1875
    XL4_3 = (10.0, 7.5)  # 3000 × 2250
    XXL4_3 = (11.7, 8.8)  # 3500 × 2625
    XXXL4_3 = (13.3, 10.0)  # 4000 × 3000
    ENORMOUS4_3 = (16.7, 12.5)  # 5000 × 3750
    XE4_3 = (33.3, 25.0)  # 10000 × 7500
