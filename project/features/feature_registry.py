from features import iris_v1_features_v1

FEATURE_REGISTRY = {
    "iris_v1_features_v1" : iris_v1_features_v1.get_features
}

def load_get_features(feature_version):
    """
    Given feature_version returns get_features function 
    """
    return FEATURE_REGISTRY[feature_version]
