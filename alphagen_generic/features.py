from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType


high = High = HIGH = Feature(FeatureType.HIGH)
low = Low = LOW = Feature(FeatureType.LOW)
volume = Volume = VOLUME = Feature(FeatureType.VOLUME)
open_ = Open = OPEN = Feature(FeatureType.OPEN)
close = Close = CLOSE = Feature(FeatureType.CLOSE)
vwap = Vwap = VWAP = Feature(FeatureType.VWAP)
target = Ref(close, -20) / close - 1
