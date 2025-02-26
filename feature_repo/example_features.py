
from datetime import timedelta
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from google.protobuf.duration_pb2 import Duration

house_source = FileSource(
    path="data/processed/house_features.parquet",
    event_timestamp_column="event_timestamp",
    )

house = Entity(
    name="house_id",
    value_type=ValueType.INT64,
    description="House identifier",
    )

house_features = FeatureView(
    name="house_features",
    entities=["house_id"],
    ttl=timedelta(days=365),
    features=[
            Feature(name="lot_area", dtype=ValueType.FLOAT),
            Feature(name="overall_qual", dtype=ValueType.FLOAT),
            Feature(name="overall_cond", dtype=ValueType.FLOAT),
            Feature(name="year_built", dtype=ValueType.FLOAT),
        ],
        input=house_source,
)
