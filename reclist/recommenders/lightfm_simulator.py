from reclist.abstractions import RecModel
from reclist.utils.config import (load_ndjson_from_bucket, PCA, VECTORS_FILE_PATH)
from reclist.utils.vectorise_sounds_data import generate_metadata_vectors


class BBCSoundsLightFMSimulatorModel(RecModel):
    """
    LightFM implementation for BBC Sounds Dataset
    The model is not trained here, only predictions are being loaded directly from the specified GCP bucket.
    """
    model_name = "lightfm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vectors_dict = None
        self.enriched_items = None
        self.vectors_type = None

    def predict(self, prediction_input, *args, **kwargs):
        """
        Predicts the top 10 similar resource IDs recommended for each user according
        to the resource IDs that they've watched

        :param prediction_input: a list of lists containing a dictionary for
                                 each resource ID watched by that user
        :return:
        """
        all_predictions = []

        return all_predictions

    def load_vectors(self, enriched_items=None, vectors_file_path=VECTORS_FILE_PATH):
        """
        Loads a resource_id:vector file into a dict.
        """
        if vectors_file_path:
            print('Loading BERT embeddings...')
            with open(vectors_file_path, 'r') as f:
                enriched_items = load_ndjson_from_bucket(filename=vectors_file_path)
                vectors_dict = {item['resourceId']: item['medium_embeddings'] for item in enriched_items}
            self.vectors_type = 'BERT synopses'
        else:
            vectors_dict = generate_metadata_vectors(enriched_items, pca=PCA)
            self.vectors_type = 'Genre+MB vectors'
        return vectors_dict

    def get_vector(self, pid):

        if not self.vectors_dict:
            print("No vectors available yet.")
            self.vectors_dict = self.load_vectors(enriched_items=self.enriched_items)

        return self.vectors_dict.get(pid, [])
