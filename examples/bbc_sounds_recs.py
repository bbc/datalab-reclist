import os
import json

from reclist.datasets import BBCSoundsSampledDataset
from reclist.recommenders.lightfm_simulator import BBCSoundsLightFMSimulatorModel
from reclist.reclist import BBCSoundsRecList
from reclist.utils.config import (get_cache_directory, download_file, load_ndjson_from_bucket, load_predictions,
                                  BBC_SOUNDS_PREDICTIONS,
                                  BBC_SOUNDS_METADATA)

RESOURCE_METADATA_PATH = '210930_all_items_enriched.ndjson'


def format_predictions(predictions):
    predictions_list = []
    for user_id, resource_ids in predictions.items():
        user_predictions = []
        for resource_id in resource_ids[0:12]:
            user_predictions.append({'userId_enc': user_id, 'resourceId': resource_id})
        predictions_list.append(user_predictions)
    return predictions_list


if __name__ == "__main__":

    sounds_dataset = BBCSoundsSampledDataset()

    model = BBCSoundsLightFMSimulatorModel()

    # load predictions
    load_from_cache = False
    cache_dir = get_cache_directory()
    predictions_filepath = os.path.join(cache_dir, "predictions.ndjson")

    if load_from_cache:
        print('Loading predictions from cache...')
        with open(os.path.join(cache_dir, predictions_filepath)) as f:
            predictions = [json.loads(line) for line in f]
    else:
        print('Loading predictions from: ', BBC_SOUNDS_PREDICTIONS)
        # load from the bucket
        predictions, _ = load_predictions(filename=BBC_SOUNDS_PREDICTIONS)
        # and save it locally
        if not os.path.exists(predictions_filepath):
            download_file(BBC_SOUNDS_PREDICTIONS, predictions_filepath)
        print('Predictions saved locally.')

    formatted_predictions = format_predictions(predictions)

    # load resource metadata
    load_metadata_from_cache = False
    metadata_filepath = os.path.join(cache_dir, RESOURCE_METADATA_PATH)
    resource_metadata = None

    if load_from_cache:
        print('Loading metadata from cache...')
        with open(metadata_filepath) as f:
            resource_metadata = [json.loads(line) for line in f]
    else:
        print('Loading metadata from: ', BBC_SOUNDS_METADATA)
        # load from the bucket
        resource_metadata = load_ndjson_from_bucket(filename=BBC_SOUNDS_METADATA)
        # and save it locally
        if not os.path.exists(metadata_filepath):
            # download_file(BBC_SOUNDS_METADATA, metadata_filepath)
            with open(metadata_filepath, 'w') as f:
                for item in resource_metadata:
                    f.write(f'{item}\n')
        print('Metadata saved locally.')

    rec_list = BBCSoundsRecList(
        model=model,
        dataset=sounds_dataset,
        y_preds=formatted_predictions,
        resource_metadata=resource_metadata
    )

    rec_list(verbose=True)
