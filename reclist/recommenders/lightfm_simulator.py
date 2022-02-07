import json

from reclist.abstractions import RecModel
from reclist.utils.vectorise_sounds_data import generate_metadata_vectors

PCA = True
VECTORS_FILE_PATH = None  # "/Users/piscoa01/Downloads/bbc-datalab-sounds-pesquet_dataset_2021-09_item_metadata_medium_embeddings.ndjson"


class BBCSoundsLightFMSimulatorModel(RecModel):
    """
    LightFM implementation for BBC Sounds Dataset
    The model is not trained here, only predictions are being loaded directly from the specified GCP bucket.
    """
    model_name = "lightfm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vectors_dict = None

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

    def load_vectors(self, vectors_file_path=VECTORS_FILE_PATH):
        """
        Loads a resource_id:vector file into a dict.
        @return:
        """
        if vectors_file_path:
            print('Loading BERT embeddings...')
            with open(vectors_file_path, 'r') as f:
                enriched_items = [json.loads(line) for line in f]
                vectors_dict = {item['resourceId']: item['medium_embeddings'] for item in enriched_items}
        else:
            vectors_dict = generate_metadata_vectors(pca=PCA)
        return vectors_dict

    def get_vector(self, pid):
        print("No vectors available yet.")
        if not self.vectors_dict:
            self.vectors_dict = self.load_vectors()

        return self.vectors_dict.get(pid, [])
