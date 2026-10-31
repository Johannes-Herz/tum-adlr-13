from zipfile import ZipFile

from tactex.settings import get_settings


def main() -> None:
    settings = get_settings()
    zip_dir = settings.data_root / "raw" / "shapenet"
    extract_dir = settings.data_root / "extracted" / "shapenet"
    extract_dir.mkdir(parents=True, exist_ok=True)

    for zip_path in zip_dir.glob("*.zip"):
        print(f"Extracting {zip_path}")
        with ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)


if __name__ == "__main__":
    main()
