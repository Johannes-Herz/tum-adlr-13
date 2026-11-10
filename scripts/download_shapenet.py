from huggingface_hub import snapshot_download

from tactex.settings import get_settings


def main() -> None:
    settings = get_settings()
    if settings.hf_token is None:
        raise ValueError("TACTEX_HF_TOKEN is missing in environment")
    download_dir = settings.data_root / "raw" / "shapenet"
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ShapeNetCore dataset to: {download_dir}")

    snapshot_download(
        repo_id="ShapeNet/ShapeNetCore",
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns="02773838.zip",
        token=settings.hf_token.get_secret_value(),
    )

    print("Download complete.")


if __name__ == "__main__":
    main()
