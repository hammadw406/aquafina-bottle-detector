"""
Stage 1 helper: bulk-download candidate images to seed the dataset.

Usage:
    python src/data/scrape_images.py --query "aquafina bottle" --limit 100 --out data/raw/positive
    python src/data/scrape_images.py --query "dasani water bottle" --limit 60 --out data/raw/negative

NOTE: This pulls from the web — always manually review/filter results before
labeling (duplicates, watermarked stock photos, and irrelevant hits are common).
Respect image licensing if you plan to redistribute the dataset publicly.

Requires: pip install icrawler
"""
import argparse
from pathlib import Path

from icrawler.builtin import BingImageCrawler


def download_images(query: str, limit: int, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    crawler = BingImageCrawler(storage={"root_dir": str(out_path)})
    crawler.crawl(keyword=query, max_num=limit)
    print(f"Downloaded up to {limit} images for '{query}' -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Download candidate images for dataset seeding")
    parser.add_argument("--query", required=True, help="Search query, e.g. 'aquafina bottle'")
    parser.add_argument("--limit", type=int, default=100, help="Max images to fetch")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    download_images(args.query, args.limit, args.out)


if __name__ == "__main__":
    main()
