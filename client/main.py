from __future__ import annotations

import argparse

from client.main_client import BiometricClient


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--server-url", required=True)
	parser.add_argument("--model-path", default="")
	parser.add_argument("--image", required=True)
	parser.add_argument("--mode", choices=["enroll", "authenticate"], required=True)
	parser.add_argument("--uuid", default="")
	parser.add_argument("--threshold", type=float, default=1.0)
	args = parser.parse_args()

	client = BiometricClient(
		server_url=args.server_url,
		model_path=args.model_path or None,
		threshold=args.threshold,
	)

	if args.mode == "enroll":
		if not args.uuid:
			raise ValueError("--uuid is required for enroll")
		result = client.enroll_user(args.image, args.uuid)
		print(result)
		return

	winner = client.authenticate_user(args.image)
	print({"winner_uuid": winner})


if __name__ == "__main__":
	main()
