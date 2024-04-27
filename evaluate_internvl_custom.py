import base64
import io
import logging
import os
import queue
import threading
import time

import dotenv
import PIL.Image
import requests
import tqdm

from custom_dataset import CustomDataset

IMAGE_DIR = "data/custom/images"
QA_FILE = "data/custom/qa.json"

CONCURRENT_REQUESTS = 16


def main() -> None:
    dotenv.load_dotenv()

    api_url = os.getenv("INTERNVL_API_URL")
    assert api_url is not None, "INTERNVL_API_URL is not set"

    dataset = CustomDataset(
        qa_file_path=QA_FILE,
        image_dir=IMAGE_DIR,
    )

    entry_queue = queue.Queue(2 * CONCURRENT_REQUESTS)
    result_queue = queue.Queue()

    progress_bar = tqdm.tqdm(total=len(dataset))
    correct_count = 0
    wrong_count = 0

    thread_list = [
        threading.Thread(
            target=process_entry,
            args=(api_url, entry_queue, result_queue),
        )
        for _ in range(CONCURRENT_REQUESTS)
    ]
    for thread in thread_list:
        thread.start()

    for index in range(len(dataset)):
        while entry_queue.full():
            time.sleep(0.1)

        entry = dataset[index]
        entry_queue.put((index, entry))

        while not result_queue.empty():
            is_answer_correct = result_queue.get()
            if is_answer_correct:
                correct_count += 1
            else:
                wrong_count += 1

            progress_bar.update()
            progress_bar.set_postfix(
                correctness=f"{correct_count}/{correct_count+wrong_count} ({correct_count / (correct_count+wrong_count):.2%})",
            )

    for thread in thread_list:
        thread.join()

    progress_bar.close()

    accuracy = correct_count / (correct_count + wrong_count)
    logging.info(f"Accuracy: {accuracy:.2%}")


def check_answer(predicted_answer: str, answer: str) -> bool:
    if predicted_answer.lower().find(answer.lower()) != -1:
        return True

    return False


def encode_image_to_base64(image: PIL.Image.Image):
    image_rgb = image.convert("RGB")
    buffered = io.BytesIO()
    image_rgb.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_entry(api_url: str, entry_queue: queue.Queue, result_queue: queue.Queue):
    time.sleep(0.1)
    while not entry_queue.empty():
        index, entry = entry_queue.get()
        image = entry["image"]
        question = entry["question"]
        answer = entry["answer"]

        image_base64 = encode_image_to_base64(image)
        question_prompt = f"{question}"

        data = {
            "question": question_prompt,
            "image": image_base64,
            "temperature": 0.8,
            "top_p": 0.7,
            "max_num": 12,
            "max_new_tokens": 1024,
            "do_sample": False,
        }

        is_answer_got = False
        while not is_answer_got:
            try:
                response = requests.post(api_url, json=data)
                response_json = response.json()
                predicted_answer = response_json["answer"]
                is_answer_got = True

            except Exception as e:
                logging.error(f"Failed to get answer: {e}")

        is_answer_correct = check_answer(predicted_answer, answer)
        result_queue.put(is_answer_correct)


if __name__ == "__main__":
    main()
