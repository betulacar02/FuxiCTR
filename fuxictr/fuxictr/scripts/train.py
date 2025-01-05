from fuxictr.model_zoo.deepfm_dcn import DeepFMDCN
from fuxictr.utils import load_dataset

if __name__ == "__main__":
    config = load_config("configs/deepfm_dcn_config.yaml")
    dataset = load_dataset(config['dataset_id'])
    
    feature_sizes = dataset.feature_sizes
    model = DeepFMDCN(feature_sizes, config['embedding_dim'], config['num_cross_layers']).build_model()

    X_train, y_train, X_val, y_val = dataset.load_data()
    model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        batch_size=config['batch_size'], 
        epochs=config['epochs']
    )
