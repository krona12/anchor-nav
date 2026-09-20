#!/usr/bin/env python3
"""Fetch the pinned MTU3D inference assets using Python 3.8+ and curl.

Paths in assets-manifest.json are relative to the repository root. Existing
verified files (including Hugging Face cache symlinks) are reused. Completed
files are never replaced when corrupt: move them aside explicitly and rerun.
Repository annotations, when listed in the manifest, are verified before any
downloads or extraction; missing or corrupt annotations stop preparation.
Only the public models are downloaded by default; licensed HM3D archives must
be supplied separately. No Python packages, credentials, or GPU are needed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile


EXAMPLE_SCENE = '00861-GLAQ4DNUx5U'
CHUNK_BYTES = 1024 * 1024


class AssetError(Exception):
    """An actionable asset validation or preparation failure."""


def relative_path(value):
    """Accept portable relative names without archive/path traversal."""
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise AssetError('Unsafe relative path: %r' % value)
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or ':' in path.parts[0]:
        raise AssetError('Unsafe path; expected a repository-relative path: %r' % value)
    return path


def validate_assets(assets, require_source=False):
    seen = set()
    for asset in assets:
        path = str(relative_path(asset['path']))
        if path in seen:
            raise AssetError('Duplicate manifest destination: %s' % path)
        seen.add(path)
        size = asset.get('bytes')
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise AssetError('Missing or invalid byte count for %s' % path)
        digest = asset.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise AssetError('Missing or invalid SHA256 for %s' % path)
        if require_source and not asset.get('source', '').startswith(('https://', 'http://')):
            raise AssetError('Missing HTTP(S) source for %s' % path)
        if 'archive_member' in asset:
            relative_path(asset['archive_member'])


def integrity_problem(path, asset):
    if not path.is_file():
        return 'not a regular file (or a broken symlink)'
    actual_size = path.stat().st_size
    if actual_size != asset['bytes']:
        return 'size mismatch: expected %d bytes, found %d' % (asset['bytes'], actual_size)
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b''):
            digest.update(chunk)
    if digest.hexdigest() != asset['sha256']:
        return 'SHA256 mismatch: expected %s, found %s' % (asset['sha256'], digest.hexdigest())
    return None


def existing_valid(root, asset):
    path = root / asset['path']
    if not os.path.lexists(str(path)):
        return False
    problem = integrity_problem(path, asset)
    if problem:
        raise AssetError('%s: %s. Existing file preserved; move it aside and rerun. '
                         'For a symlink, inspect its target before changing anything.' % (path, problem))
    print('Verified: %s' % asset['path'], flush=True)
    return True


def writable_destination(root, asset):
    path = root / asset['path']
    # Valid external cache symlinks are accepted by existing_valid(), but new
    # downloads/extractions must never write through them outside this root.
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise AssetError('Refusing to write outside repository root: %s. '
                         'Move the destination symlink aside and rerun.' % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def promote(partial, destination, root, asset):
    problem = integrity_problem(partial, asset)
    if problem:
        raise AssetError('%s: %s. Move this partial file aside and rerun for a fresh transfer.' %
                         (partial, problem))
    # Recheck rather than replacing a completed file another process created.
    if existing_valid(root, asset):
        return
    os.replace(str(partial), str(destination))
    print('Installed: %s' % asset['path'], flush=True)


def download(root, asset):
    if existing_valid(root, asset):
        return
    destination = writable_destination(root, asset)
    partial = destination.with_name(destination.name + '.part')
    if os.path.lexists(str(partial)):
        if partial.is_symlink() or not partial.is_file():
            raise AssetError('Unsafe partial file: %s. Move it aside and rerun.' % partial)
        if partial.stat().st_size >= asset['bytes']:
            promote(partial, destination, root, asset)
            return
    if shutil.which('curl') is None:
        raise AssetError('curl is required for downloads; install curl and rerun.')
    print('Downloading: %s (resumes %s)' % (asset['path'], partial.name), flush=True)
    command = [
        'curl', '--fail', '--location', '--silent', '--show-error',
        '--retry', '5', '--retry-delay', '2', '--retry-connrefused',
        '--connect-timeout', '30', '--continue-at', '-',
        '--proto', '=http,https', '--proto-redir', '=http,https',
        '--output', str(partial), '--url', asset['source'],
    ]
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise AssetError('curl failed with exit code %d for %s. The .part file is preserved; '
                         'rerun to resume. If the server refuses range requests, move %s '
                         'aside and rerun for a fresh download.' %
                         (result.returncode, asset['path'], partial))
    promote(partial, destination, root, asset)


def extract_scenes(root, archive_path, assets):
    """Extract only allowlisted regular files, never archive-controlled paths."""
    missing = [asset for asset in assets if not existing_valid(root, asset)]
    if not missing:
        return
    required = {str(relative_path(asset['archive_member'])): asset for asset in missing}
    if len(required) != len(missing):
        raise AssetError('Duplicate required archive members in manifest')
    with tarfile.open(str(archive_path), 'r:*') as archive:
        selected = {}
        # Inspect the complete directory first so a later duplicate or unsafe
        # member cannot be discovered only after we have installed files.
        for member in archive.getmembers():
            name = str(relative_path(member.name))
            if name not in required:
                continue
            if name in selected:
                raise AssetError('Duplicate required archive member: %s' % name)
            if not member.isfile():
                raise AssetError('Required archive member must be a regular file: %s' % name)
            if member.size != required[name]['bytes']:
                raise AssetError('Archive size mismatch for %s: expected %d, found %d' %
                                 (name, required[name]['bytes'], member.size))
            selected[name] = member
        absent = sorted(set(required) - set(selected))
        if absent:
            raise AssetError('Archive is missing required meshes: %s' % ', '.join(absent))
        for name, asset in required.items():
            destination = writable_destination(root, asset)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=destination.name + '.', suffix='.part', dir=str(destination.parent))
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, 'wb') as output:
                    with archive.extractfile(selected[name]) as source:
                        shutil.copyfileobj(source, output, CHUNK_BYTES)
                promote(temporary, destination, root, asset)
            finally:
                if temporary.exists():
                    temporary.unlink()
    print('Scene archive preserved: %s' % archive_path, flush=True)


def example_asset(hm3d):
    assets = [asset for asset in hm3d['files']
              if PurePosixPath(asset['path']).parent.name == EXAMPLE_SCENE]
    if len(assets) != 1:
        raise AssetError('Manifest must include exactly one public example mesh for %s' % EXAMPLE_SCENE)
    return assets[0]


def prepare_example(root, hm3d):
    scene = example_asset(hm3d)
    if existing_valid(root, scene):
        return
    if not all(hm3d.get('public_example_archive_' + field) is not None
               for field in ('source', 'bytes', 'sha256')):
        raise AssetError('Manifest lacks the public example archive source, byte count, or SHA256.')
    archive_asset = {
        'path': '.cache/mtu3d-assets/hm3d-example-habitat-v0.2.tar',
        'source': hm3d['public_example_archive_source'],
        'bytes': hm3d['public_example_archive_bytes'],
        'sha256': hm3d['public_example_archive_sha256'],
    }
    validate_assets([archive_asset], require_source=True)
    download(root, archive_asset)
    extract_scenes(root, root / archive_asset['path'], [scene])


def verify_all(root, assets):
    problems = []
    for asset in assets:
        try:
            if not existing_valid(root, asset):
                problems.append('Missing: %s' % asset['path'])
        except AssetError as error:
            problems.append(str(error))
    if problems:
        raise AssetError('\n'.join(problems))


def main(argv=None):
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root', type=Path, default=default_root,
                        help='destination repository root (default: parent of this script directory)')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('assets-manifest.json'),
                        help='asset manifest (default: beside this script)')
    parser.add_argument('--verify-only', action='store_true',
                        help='check files without downloads, extraction, directory creation, or other writes')
    parser.add_argument('--hm3d-archive', type=Path, metavar='PATH',
                        help='extract only the 36 required meshes from your authorized HM3D habitat tar archive')
    parser.add_argument('--require-scenes', action='store_true',
                        help='also require all 36 HM3D meshes for a full benchmark')
    parser.add_argument('--example-scene', action='store_true',
                        help='also prepare/check public scene 00861-GLAQ4DNUx5U; insufficient for full benchmark')
    args = parser.parse_args(argv)
    if args.verify_only and args.hm3d_archive:
        parser.error('--verify-only cannot be combined with --hm3d-archive (which extracts files)')
    try:
        root = args.repo_root.resolve()
        with args.manifest.open(encoding='utf-8') as source:
            manifest = json.load(source)
        annotations = manifest.get('annotations', {}).get('files', [])
        validate_assets(annotations)
        verify_all(root, annotations)
        models = manifest['model_files']
        hm3d = manifest['hm3d']
        scenes = hm3d['files']
        validate_assets(models, require_source=True)
        validate_assets(scenes)
        selected_scenes = scenes if args.require_scenes else ([example_asset(hm3d)] if args.example_scene else [])
        if args.verify_only:
            verify_all(root, models + selected_scenes)
        else:
            for asset in models:
                download(root, asset)
            if args.hm3d_archive:
                extract_scenes(root, args.hm3d_archive, scenes)
            if args.example_scene:
                prepare_example(root, hm3d)
            if args.require_scenes:
                verify_all(root, scenes)
        print('Ready: %d model files%s.' %
              (len(models), ', %d requested scene meshes' % len(selected_scenes) if selected_scenes else ''))
        if not args.require_scenes:
            print('Full benchmark scene availability was not checked; use --verify-only --require-scenes.')
        return 0
    except (AssetError, OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        print('ERROR: %s' % error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
