"""CLI integration checks: real curl against a local range-capable HTTP server."""
import hashlib
import http.server
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest

SCRIPT = Path(__file__).with_name('download_assets.py')
PAYLOAD = b'verified payload\n' * 1024
SCENE = '00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb'

class Handler(http.server.BaseHTTPRequestHandler):
    data = PAYLOAD
    ranges = []
    def do_GET(self):
        start = 0
        header = self.headers.get('Range')
        self.ranges.append(header)
        if header:
            start = int(header.split('=')[1].split('-')[0])
        self.send_response(206 if header else 200)
        self.send_header('Content-Length', str(len(self.data) - start))
        if header:
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, len(self.data)-1, len(self.data)))
        self.end_headers()
        self.wfile.write(self.data[start:])
    def log_message(self, *args):
        pass

class DownloaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:%d/model' % cls.server.server_port
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mtu3d-assets-test-')
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.json'
        Handler.data = PAYLOAD
        Handler.ranges = []
        self.model = self.asset('checkpoint/model.bin', PAYLOAD, source=self.url)
        self.mesh = self.asset('datascene/' + SCENE, b'mesh data', archive_member=SCENE)
        self.data = {'model_files': [self.model], 'hm3d': {'files': [self.mesh]}}
        self.save()
    def tearDown(self):
        self.temp.cleanup()
    def asset(self, path, payload, **extra):
        return dict(path=path, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), **extra)
    def save(self):
        self.manifest.write_text(json.dumps(self.data))
    def write(self, path, content):
        dest = self.root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return dest
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), '--repo-root', str(self.root), '--manifest', str(self.manifest), *args], text=True, capture_output=True)
    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    def ready_model(self):
        return self.write(self.model['path'], PAYLOAD)
    def archive(self, members):
        path = self.root / 'scenes.tar'
        with tarfile.open(str(path), 'w') as out:
            for name, content, kind in members:
                info = tarfile.TarInfo(name)
                info.type = kind
                if kind == tarfile.REGTYPE:
                    info.size = len(content)
                    out.addfile(info, io.BytesIO(content))
                else:
                    info.linkname = content.decode()
                    out.addfile(info)
        return path
    def test_invalid_annotations_block_downloads_and_extraction(self):
        annotation = self.asset('LangMap_Annotations/scene.json.gz', b'good')
        self.data['annotations'] = {'files': [annotation]}
        self.save()
        archive = self.archive([(SCENE, b'mesh data', tarfile.REGTYPE)])
        for content in (None, b'wrong-size', b'baad'):
            with self.subTest(content=content):
                if content is not None:
                    self.write(annotation['path'], content)
                before = {str(p.relative_to(self.root)): p.read_bytes() if p.is_file() else None
                          for p in self.root.rglob('*')}
                result = self.run_cli('--hm3d-archive', str(archive))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(annotation['path'], result.stderr)
                after = {str(p.relative_to(self.root)): p.read_bytes() if p.is_file() else None
                         for p in self.root.rglob('*')}
                self.assertEqual(after, before)
                self.assertEqual(Handler.ranges, [])
    def test_valid_annotations_allow_model_preparation(self):
        annotation = self.asset('LangMap_Annotations/scene.json.gz', b'good')
        self.data['annotations'] = {'files': [annotation]}
        self.save()
        self.write(annotation['path'], b'good')
        self.assert_ok(self.run_cli())
        self.assertEqual((self.root / self.model['path']).read_bytes(), PAYLOAD)
    def test_fresh_download_and_idempotent_skip(self):
        self.assert_ok(self.run_cli())
        dest = self.root / self.model['path']
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        original = dest.stat().st_mtime_ns
        self.assert_ok(self.run_cli())
        self.assertEqual(dest.stat().st_mtime_ns, original)
        self.assertEqual(len(Handler.ranges), 1)
        self.assertFalse(dest.with_name(dest.name + '.part').exists())
    def test_partial_download_resumes_with_real_range_request(self):
        self.write(self.model['path'] + '.part', PAYLOAD[:123])
        self.assert_ok(self.run_cli())
        self.assertEqual((self.root / self.model['path']).read_bytes(), PAYLOAD)
        self.assertEqual(Handler.ranges, ['bytes=123-'])
    def test_corrupt_completed_file_is_preserved(self):
        bad = b'x' * len(PAYLOAD)
        dest = self.write(self.model['path'], bad)
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256', result.stderr)
        self.assertIn('aside', result.stderr)
        self.assertEqual(dest.read_bytes(), bad)
        self.assertEqual(Handler.ranges, [])
    def test_download_hash_mismatch_is_never_promoted(self):
        Handler.data = b'x' * len(PAYLOAD)
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256', result.stderr)
        self.assertFalse((self.root / self.model['path']).exists())
        self.assertEqual((self.root / (self.model['path'] + '.part')).read_bytes(), Handler.data)
    def test_verify_only_missing_has_no_side_effects(self):
        before = sorted(str(p) for p in self.root.rglob('*'))
        result = self.run_cli('--verify-only', '--require-scenes')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(self.model['path'], result.stderr)
        self.assertIn(self.mesh['path'], result.stderr)
        self.assertEqual(sorted(str(p) for p in self.root.rglob('*')), before)
        self.assertEqual(Handler.ranges, [])
    def test_verify_only_valid_symlinked_models(self):
        cache = self.write('external-cache/weights.bin', PAYLOAD)
        target = self.root / self.model['path']
        target.parent.mkdir()
        target.symlink_to(cache)
        self.write(self.mesh['path'], b'mesh data')
        self.assert_ok(self.run_cli('--verify-only', '--require-scenes'))
        self.assertTrue(target.is_symlink())
        self.assertEqual(Handler.ranges, [])
    def test_extracts_only_required_mesh_and_preserves_archive(self):
        self.ready_model()
        archive = self.archive([(SCENE, b'mesh data', tarfile.REGTYPE), ('other/other.basis.glb', b'extra', tarfile.REGTYPE), ('other/navmesh', b'extra', tarfile.REGTYPE)])
        before = archive.read_bytes()
        self.assert_ok(self.run_cli('--hm3d-archive', str(archive), '--require-scenes'))
        self.assertEqual((self.root / self.mesh['path']).read_bytes(), b'mesh data')
        self.assertFalse((self.root / 'datascene/other').exists())
        self.assertEqual(archive.read_bytes(), before)
    def test_extraction_hash_mismatch_is_never_promoted(self):
        self.ready_model()
        archive = self.archive([(SCENE, b'wrongdata', tarfile.REGTYPE)])
        result = self.run_cli('--hm3d-archive', str(archive))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256', result.stderr)
        self.assertFalse((self.root / self.mesh['path']).exists())
        self.assertTrue(archive.exists())
    def test_rejects_archive_traversal_before_extracting(self):
        self.ready_model()
        archive = self.archive([(SCENE, b'mesh data', tarfile.REGTYPE), ('../escape', b'no', tarfile.REGTYPE)])
        result = self.run_cli('--hm3d-archive', str(archive))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unsafe', result.stderr)
        self.assertFalse((self.root / self.mesh['path']).exists())
    def test_rejects_selected_archive_link(self):
        self.ready_model()
        archive = self.archive([(SCENE, b'/tmp/escape', tarfile.SYMTYPE)])
        result = self.run_cli('--hm3d-archive', str(archive))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('regular', result.stderr)
        self.assertFalse((self.root / self.mesh['path']).exists())
    def test_rejects_duplicate_selected_member(self):
        self.ready_model()
        archive = self.archive([(SCENE, b'mesh data', tarfile.REGTYPE), (SCENE, b'mesh data', tarfile.REGTYPE)])
        result = self.run_cli('--hm3d-archive', str(archive))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Duplicate', result.stderr)
        self.assertFalse((self.root / self.mesh['path']).exists())
    def test_rejects_symlink_extraction_parent(self):
        self.ready_model()
        outside = self.root / 'outside'
        outside.mkdir()
        (self.root / 'datascene').symlink_to(outside.parent.parent)
        archive = self.archive([(SCENE, b'mesh data', tarfile.REGTYPE)])
        result = self.run_cli('--hm3d-archive', str(archive))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('outside', result.stderr)
    def test_rejects_absolute_manifest_destination(self):
        self.model['path'] = '/tmp/mtu3d-should-not-write'
        self.save()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('relative', result.stderr)
        self.assertEqual(Handler.ranges, [])
    def test_example_download_extracts_only_requested_scene(self):
        self.ready_model()
        archive = self.archive([(SCENE, b'mesh data', tarfile.REGTYPE), ('other/other.glb', b'ignore', tarfile.REGTYPE)])
        Handler.data = archive.read_bytes()
        self.data['hm3d'].update(public_example_archive_source=self.url,
                                 public_example_archive_bytes=len(Handler.data),
                                 public_example_archive_sha256=hashlib.sha256(Handler.data).hexdigest())
        self.save()
        self.assert_ok(self.run_cli('--example-scene'))
        self.assertEqual((self.root / self.mesh['path']).read_bytes(), b'mesh data')
        self.assertFalse((self.root / 'datascene/other').exists())
        self.assertEqual((self.root / '.cache/mtu3d-assets/hm3d-example-habitat-v0.2.tar').read_bytes(), Handler.data)
    def test_complete_partial_is_promoted_without_network(self):
        self.write(self.model['path'] + '.part', PAYLOAD)
        self.assert_ok(self.run_cli())
        self.assertEqual((self.root / self.model['path']).read_bytes(), PAYLOAD)
        self.assertEqual(Handler.ranges, [])
    def test_corrupt_complete_partial_is_preserved_without_network(self):
        partial = self.write(self.model['path'] + '.part', b'x' * len(PAYLOAD))
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256', result.stderr)
        self.assertEqual(partial.read_bytes(), b'x' * len(PAYLOAD))
        self.assertEqual(Handler.ranges, [])
    def test_partial_symlink_is_not_followed(self):
        outside = self.write('innocent', b'do not change')
        partial = self.root / (self.model['path'] + '.part')
        partial.parent.mkdir()
        partial.symlink_to(outside)
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unsafe partial', result.stderr)
        self.assertEqual(outside.read_bytes(), b'do not change')
        self.assertEqual(Handler.ranges, [])
    def test_corrupt_existing_mesh_is_preserved(self):
        self.ready_model()
        mesh = self.write(self.mesh['path'], b'wrongdata')
        archive = self.archive([(SCENE, b'mesh data', tarfile.REGTYPE)])
        result = self.run_cli('--hm3d-archive', str(archive))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256', result.stderr)
        self.assertEqual(mesh.read_bytes(), b'wrongdata')
    def test_archive_missing_mesh_fails_before_writing(self):
        self.ready_model()
        archive = self.archive([('unneeded', b'x', tarfile.REGTYPE)])
        result = self.run_cli('--hm3d-archive', str(archive))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing required', result.stderr)
        self.assertFalse((self.root / 'datascene').exists())
    def test_example_scene_skips_download_when_valid(self):
        self.ready_model()
        self.write(self.mesh['path'], b'mesh data')
        self.assert_ok(self.run_cli('--example-scene'))
        self.assertEqual(Handler.ranges, [])

if __name__ == '__main__':
    unittest.main(verbosity=2)
