import bz2
import logging
import os
import sys
import time
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
FILENAME = "shape_predictor_68_face_landmarks.dat.bz2"
OUTPUT_FILE = "shape_predictor_68_face_landmarks.dat"
CHUNK_SIZE = 1024 * 1024
RETRIES = 3
TIMEOUT_SECONDS = 60


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("dlib_downloader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def _print_progress(downloaded: int, total: int | None, start_time: float) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    speed = downloaded / elapsed
    if total:
        percent = (downloaded / total) * 100
        msg = (
            f"\rDownloading: {percent:6.2f}% "
            f"({_format_size(downloaded)}/{_format_size(total)}) "
            f"{_format_size(int(speed))}/s"
        )
    else:
        msg = f"\rDownloading: {_format_size(downloaded)} {_format_size(int(speed))}/s"
    print(msg, end="", flush=True)


def download_with_retry(url: str, destination: str, logger: logging.Logger) -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; dlib-downloader/1.0)"}
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            logger.info("Downloading file (attempt %d/%d)...", attempt, RETRIES)
            request = Request(url, headers=headers)

            with urlopen(request, timeout=TIMEOUT_SECONDS) as response, open(destination, "wb") as out_file:
                total = response.length
                downloaded = 0
                start_time = time.time()

                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    out_file.flush()
                    downloaded += len(chunk)
                    _print_progress(downloaded, total, start_time)

            print()
            logger.info("Download selesai: %s", destination)
            return

        except (URLError, HTTPError, TimeoutError, ConnectionResetError, IncompleteRead, OSError) as exc:
            last_error = exc
            print()
            logger.error("Download gagal pada attempt %d: %s", attempt, exc)

            if os.path.exists(destination):
                try:
                    os.remove(destination)
                except OSError:
                    pass

            if attempt < RETRIES:
                backoff = 2 ** attempt
                logger.info("Retry dalam %d detik...", backoff)
                time.sleep(backoff)

    raise RuntimeError(f"Semua percobaan download gagal. Error terakhir: {last_error}") from last_error


def decompress_bz2(source: str, target: str, logger: logging.Logger) -> None:
    logger.info("Decompressing file...")
    written = 0
    start_time = time.time()

    with bz2.open(source, "rb") as f_in, open(target, "wb") as f_out:
        while True:
            chunk = f_in.read(CHUNK_SIZE)
            if not chunk:
                break
            f_out.write(chunk)
            f_out.flush()
            written += len(chunk)
            elapsed = max(time.time() - start_time, 1e-6)
            speed = written / elapsed
            print(
                f"\rDecompressing: {_format_size(written)} {_format_size(int(speed))}/s",
                end="",
                flush=True,
            )

    print()
    logger.info("Decompress selesai: %s", target)


def main() -> None:
    logger = setup_logger()
    try:
        download_with_retry(URL, FILENAME, logger)
        decompress_bz2(FILENAME, OUTPUT_FILE, logger)
        logger.info("Done!")
    except Exception as exc:
        logger.exception("Proses gagal: %s", exc)
        raise


if __name__ == "__main__":
    main()
