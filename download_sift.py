import os
import urllib.request
import tarfile

def download_sift():
    data_dir = "tmp"
    sift_dir = os.path.join(data_dir, "sift")
    sift_url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    sift_tar = os.path.join(data_dir, "sift.tar.gz")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(sift_dir):
        print("SIFT1M already exists.")
        return

    print(f"Downloading {sift_url}...")
    try:
        urllib.request.urlretrieve(sift_url, sift_tar)
    except Exception as e:
        # If FTP fails, maybe use a mirror? But let's try first.
        print(f"Download failed: {e}")
        return

    print("Extracting...")
    with tarfile.open(sift_tar, "r:gz") as tar:
        tar.extractall(path=data_dir)
    os.remove(sift_tar)
    print("Done.")

if __name__ == "__main__":
    download_sift()
