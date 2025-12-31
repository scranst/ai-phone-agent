"""
Batch Calling Script - Call multiple businesses and collect information

Examples:
  python batch_calls.py --search "pool maintenance las vegas" \
    --objective "Get a quote for weekly pool maintenance for a 15,000 gallon in-ground pool with salt system"

  python batch_calls.py --search "med spa glp-1 las vegas" \
    --objective "Ask about GLP-1 weight loss medication pricing. Get prices for semaglutide and tirzepatide, monthly cost, and if consultation is included"

  python batch_calls.py --file contacts.json \
    --objective "Your objective here"
"""

import asyncio
import argparse
import json
import os
import csv
from datetime import datetime
from typing import Optional
import logging

from agent_sim7600 import PhoneAgentSIM7600, CallRequest, CallResult

# Optional: for searching businesses
try:
    import googlemaps
    HAS_GOOGLEMAPS = True
except ImportError:
    HAS_GOOGLEMAPS = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def search_google_places(query: str, api_key: str, max_results: int = 20) -> list[dict]:
    """Search Google Places API for businesses"""
    if not HAS_GOOGLEMAPS:
        raise ImportError("googlemaps not installed. Run: pip install googlemaps")

    gmaps = googlemaps.Client(key=api_key)
    results = []

    # Text search for businesses
    response = gmaps.places(query=query)

    for place in response.get('results', [])[:max_results]:
        # Get place details for phone number
        details = gmaps.place(place['place_id'], fields=['formatted_phone_number', 'name', 'formatted_address'])
        detail = details.get('result', {})

        if detail.get('formatted_phone_number'):
            results.append({
                'name': detail.get('name', place.get('name', 'Unknown')),
                'phone': detail.get('formatted_phone_number'),
                'address': detail.get('formatted_address', ''),
            })

    return results


def load_contacts_from_file(filepath: str) -> list[dict]:
    """Load contacts from JSON or CSV file"""
    if filepath.endswith('.json'):
        with open(filepath) as f:
            return json.load(f)
    elif filepath.endswith('.csv'):
        contacts = []
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                contacts.append({
                    'name': row.get('name', row.get('Name', 'Unknown')),
                    'phone': row.get('phone', row.get('Phone', row.get('phone_number', ''))),
                    'address': row.get('address', row.get('Address', '')),
                })
        return contacts
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def save_results(results: list[dict], output_path: str):
    """Save results to JSON and CSV"""
    # Save JSON
    json_path = output_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # Save CSV for easy viewing
    csv_path = output_path if output_path.endswith('.csv') else output_path.replace('.json', '.csv')
    if results:
        with open(csv_path, 'w', newline='') as f:
            # Flatten collected_info into columns
            fieldnames = ['name', 'phone', 'address', 'success', 'summary', 'duration_seconds']

            # Add any keys from collected_info
            all_info_keys = set()
            for r in results:
                all_info_keys.update(r.get('collected_info', {}).keys())
            fieldnames.extend(sorted(all_info_keys))

            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for r in results:
                row = {
                    'name': r.get('name', ''),
                    'phone': r.get('phone', ''),
                    'address': r.get('address', ''),
                    'success': r.get('success', False),
                    'summary': r.get('summary', ''),
                    'duration_seconds': r.get('duration_seconds', 0),
                }
                # Add collected_info fields
                row.update(r.get('collected_info', {}))
                writer.writerow(row)

        logger.info(f"Results saved to {csv_path}")


async def run_batch_calls(
    contacts: list[dict],
    objective: str,
    context: dict = None,
    delay_between_calls: float = 5.0,
    audio_input: Optional[str] = None,
    audio_output: Optional[str] = None,
) -> list[dict]:
    """
    Call multiple contacts and collect results.

    Args:
        contacts: List of dicts with 'name', 'phone', and optionally 'address'
        objective: What to accomplish on each call
        context: Additional context for the AI
        delay_between_calls: Seconds to wait between calls
        audio_input: Audio input device name
        audio_output: Audio output device name

    Returns:
        List of results for each call
    """
    agent = PhoneAgentSIM7600(
        audio_input_device=audio_input,
        audio_output_device=audio_output
    )

    # Connect to modem
    logger.info("Connecting to SIM7600 modem...")
    if not agent.connect_modem():
        raise Exception("Failed to connect to modem")

    network = agent.get_network_info()
    logger.info(f"Network: {network.get('operator', 'Unknown')}, Signal: {network.get('signal_dbm', '?')} dBm")

    results = []
    total = len(contacts)

    for i, contact in enumerate(contacts, 1):
        name = contact.get('name', 'Unknown')
        phone = contact.get('phone', '')
        address = contact.get('address', '')

        if not phone:
            logger.warning(f"[{i}/{total}] Skipping {name} - no phone number")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{total}] Calling: {name}")
        logger.info(f"Phone: {phone}")
        if address:
            logger.info(f"Address: {address}")
        logger.info(f"{'='*60}")

        # Build context for this call
        call_context = context.copy() if context else {}
        call_context['business_name'] = name
        if address:
            call_context['business_address'] = address

        request = CallRequest(
            phone=phone,
            objective=objective,
            context=call_context
        )

        try:
            result = await agent.call(request)

            results.append({
                'name': name,
                'phone': phone,
                'address': address,
                'success': result.success,
                'summary': result.summary,
                'collected_info': result.collected_info,
                'transcript': result.transcript,
                'duration_seconds': result.duration_seconds,
                'recording_path': result.recording_path,
            })

            logger.info(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
            logger.info(f"Summary: {result.summary}")
            if result.collected_info:
                logger.info(f"Collected: {json.dumps(result.collected_info, indent=2)}")

        except Exception as e:
            logger.error(f"Call failed with error: {e}")
            results.append({
                'name': name,
                'phone': phone,
                'address': address,
                'success': False,
                'summary': f"Error: {str(e)}",
                'collected_info': {},
                'transcript': [],
                'duration_seconds': 0,
            })

        # Delay before next call
        if i < total:
            logger.info(f"Waiting {delay_between_calls}s before next call...")
            await asyncio.sleep(delay_between_calls)

    # Disconnect
    agent.disconnect_modem()

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Batch calling script - call multiple businesses and collect information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Contact sources (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--search', help='Google Places search query (requires GOOGLE_MAPS_API_KEY)')
    source.add_argument('--file', help='JSON or CSV file with contacts (name, phone, address)')
    source.add_argument('--phones', nargs='+', help='Phone numbers to call directly')

    parser.add_argument('--objective', '-o', required=True,
                       help='What to accomplish on each call')
    parser.add_argument('--context', '-c', action='append', nargs=2,
                       metavar=('KEY', 'VALUE'), help='Context key-value pairs')
    parser.add_argument('--output', '-O', default=None,
                       help='Output file path (default: results_TIMESTAMP.csv)')
    parser.add_argument('--delay', type=float, default=5.0,
                       help='Seconds between calls (default: 5)')
    parser.add_argument('--max-results', type=int, default=20,
                       help='Max results from search (default: 20)')
    parser.add_argument('--audio-in', help='Audio input device name')
    parser.add_argument('--audio-out', help='Audio output device name')
    parser.add_argument('--dry-run', action='store_true',
                       help='List contacts without calling')

    args = parser.parse_args()

    # Get contacts
    contacts = []

    if args.search:
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            print("ERROR: GOOGLE_MAPS_API_KEY environment variable required for --search")
            print("Get one at: https://console.cloud.google.com/apis/credentials")
            return

        print(f"Searching for: {args.search}")
        contacts = search_google_places(args.search, api_key, args.max_results)
        print(f"Found {len(contacts)} businesses with phone numbers")

    elif args.file:
        contacts = load_contacts_from_file(args.file)
        print(f"Loaded {len(contacts)} contacts from {args.file}")

    elif args.phones:
        contacts = [{'name': f'Contact {i+1}', 'phone': p} for i, p in enumerate(args.phones)]

    if not contacts:
        print("No contacts found!")
        return

    # Show contacts
    print(f"\nContacts to call ({len(contacts)}):")
    print("-" * 60)
    for c in contacts:
        print(f"  {c.get('name', 'Unknown'):30} {c.get('phone', 'No phone')}")
    print("-" * 60)

    if args.dry_run:
        print("\n[DRY RUN] No calls made.")
        return

    # Build context
    context = {}
    if args.context:
        for key, value in args.context:
            context[key] = value

    # Confirm
    print(f"\nObjective: {args.objective}")
    if context:
        print(f"Context: {context}")

    confirm = input(f"\nProceed with {len(contacts)} calls? [y/N] ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # Run calls
    results = await run_batch_calls(
        contacts=contacts,
        objective=args.objective,
        context=context,
        delay_between_calls=args.delay,
        audio_input=args.audio_in,
        audio_output=args.audio_out,
    )

    # Save results
    output_path = args.output or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results(results, output_path)

    # Summary
    successful = sum(1 for r in results if r.get('success'))
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total calls: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
