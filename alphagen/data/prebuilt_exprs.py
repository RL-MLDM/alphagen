from alphagen.data.expression import *

CLOSE = Feature(FeatureType.CLOSE)

TARGET_5D = Ref(CLOSE, -5) / CLOSE - 1
TARGET_20D = Ref(CLOSE, -20) / CLOSE - 1
