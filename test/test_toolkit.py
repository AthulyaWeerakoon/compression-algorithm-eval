import unittest
import json
from unittest.mock import patch
import numpy as np
from compression_evaluation.evaluator import Evaluator
from compression_evaluation.classes import (PCEncoder, PCDecoder, MultivariatePCEncoder,
    MultivariatePCDecoder, UniformQuantizer, ANSCoder, SimpleFrequencyTable)

class StatefulPredictor:
    def __init__(self): self.window = [0.0, 0.0]; self.last = []
    def predict(self, n):
        result = np.full(n, self.window[-1])
        self.last = list(result)
        return result
    def update(self, residuals):
        values = np.asarray(self.last) + np.asarray(residuals)
        self.window = (self.window + values.tolist())[-2:]
class MatrixPredictor:
    def __init__(self): self.windows = np.zeros((2,2)); self.last = None
    def predict(self,n):
        self.last = np.repeat(self.windows[:,-1,None],n,axis=1)
        return self.last.copy()
    def update(self,residuals):
        values = self.last + residuals
        self.windows = np.concatenate((self.windows,values),axis=1)[:,-2:]
class ToyCoder:
    # Test-only lossless integer symbol serialization; no byte truncation.
    def encode_symbols(self,x): return json.dumps([int(v) for v in x]).encode()
    def decode_symbols(self,b): return json.loads(b.decode())

class ToolkitTests(unittest.TestCase):
    def evaluate(self, data, stream=b'12', recon=None):
        class E:
            def encode(self,x): return stream
        class D:
            def decode(self,x): return np.asarray(data if recon is None else recon)
        with patch('compression_evaluation.evaluator.profile_memory', side_effect=lambda f,x:(f(x),1.,.1)):
            return Evaluator(E(),D()).evaluate(data)

    def test_evaluator_metrics_and_old_keys(self):
        x=np.array([1,2],dtype=np.float32); m=self.evaluate(x)
        self.assertEqual(m['uncompressed_size_bytes'],x.nbytes)
        self.assertEqual(m['compressed_size_bytes'],2); self.assertEqual(m['symbol_count'],2)
        for k in ('encode_time_sec','encode_mem_mb','decode_time_sec','decode_mem_mb','compression_ratio','bits_per_symbol','mse','snr_db'): self.assertIn(k,m)
        self.assertEqual(m['bits_per_symbol'],8)
    def test_two_dimensional_and_snr(self):
        x=np.array([[1,2,3],[4,5,6]],dtype=np.int16)
        m=self.evaluate(x,recon=x+1); self.assertEqual(m['symbol_count'],6); self.assertEqual(m['uncompressed_size_bytes'],12); self.assertEqual(m['mse'],1)
        self.assertTrue(np.isfinite(m['snr_db']))
        self.assertEqual(self.evaluate(np.zeros(2),recon=np.ones(2))['snr_db'],float('-inf'))
    def test_extreme_metric_underflow_overflow_and_snr(self):
        def metric(source, reconstructed):
            return self.evaluate(np.array([source], dtype=np.float64), recon=[reconstructed])
        with self.assertRaises(ValueError): metric(1e-200, 0.0)
        result=metric(1e-160, 0.0)
        self.assertGreater(result['mse'],0.0); self.assertAlmostEqual(result['snr_db'],0.0,places=6)
        self.assertAlmostEqual(metric(1e-200,1.0)['snr_db'],-4000.0,places=5)
        result=self.evaluate(np.array([1.0,1e-160]),recon=[1.0,0.0])
        self.assertTrue(np.isfinite(result['snr_db']))
        self.assertEqual(metric(1e-200,1e-200)['snr_db'],float('inf'))
        self.assertEqual(metric(0.0,1.0)['snr_db'],float('-inf'))
        largest=np.finfo(np.float64).max
        with self.assertRaises(ValueError): metric(largest,-largest)
        with self.assertRaises(ValueError): metric(largest,largest)

    def test_multistream_payload(self):
        class E:
            def encode(self,x): return [b'a',bytearray(b'bc')]
        class D:
            def decode(self,x): self.got=x; return [1,2]
        d=D()
        with patch('compression_evaluation.evaluator.profile_memory',side_effect=lambda f,x:(f(x),0,0)):
            m=Evaluator(E(),d).evaluate([1,2])
        self.assertEqual(m['compressed_size_bytes'],3); self.assertIsInstance(d.got,list)
    def test_evaluator_rejects_empty_input_before_encoder(self):
        class E:
            def encode(self, x): raise AssertionError('encoder called')
        class D:
            def decode(self, x): raise AssertionError('decoder called')
        with self.assertRaises(ValueError): Evaluator(E(),D()).evaluate(np.array([],dtype=float))

    def test_reject_invalid_inputs_before_encoding(self):
        class E:
            def encode(self,x): raise AssertionError('called')
        class D:
            def decode(self, stream): raise AssertionError('called')
        for x in ([], 3, [[1],[2,3]], np.array([True]), np.array([1+2j]), [float('nan')], [2**60+1]):
            with self.subTest(x=x), self.assertRaises(ValueError): Evaluator(E(),D()).evaluate(x)
    def test_bad_reconstruction_and_bytearray_payload(self):
        class E:
            def encode(self, x): return bytearray(b'x')
        class D:
            def decode(self, x): return [1, float('inf')]
        with patch('compression_evaluation.evaluator.profile_memory',side_effect=lambda f,x:(f(x),0,0)):
            with self.assertRaises(ValueError): Evaluator(E(),D()).evaluate([1,2])

    def test_bad_stream_before_decode(self):
        class E:
            def encode(self,x): return []
        class D:
            def decode(self,x): raise AssertionError('called')
        with patch('compression_evaluation.evaluator.profile_memory',side_effect=lambda f,x:(f(x),0,0)):
            with self.assertRaises(ValueError): Evaluator(E(),D()).evaluate([1])
    def test_scalar_empty_stream_and_predictor_shape_rejected(self):
        class EmptyCoder(ToyCoder):
            def decode_symbols(self, stream): return []
        with self.assertRaises(ValueError): PCDecoder(StatefulPredictor(),EmptyCoder()).decode(b'[]')
        class WrongShape:
            def predict(self,n): return np.zeros((n,1))
            def update(self,x): raise AssertionError('update called')
        with self.assertRaises(ValueError): PCEncoder(WrongShape(),ToyCoder()).encode([1])
        with self.assertRaises(ValueError): PCDecoder(WrongShape(),ToyCoder()).decode(b'[1]')

    def test_scalar_rejects_float64_conversion_overflow_before_prediction(self):
        if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
            self.skipTest('longdouble has no wider range than float64')
        class NeverPredict:
            def predict(self,n): raise AssertionError('predictor called')
            def update(self,x): raise AssertionError('update called')
        value=np.array([np.finfo(np.longdouble).max],dtype=np.longdouble)
        self.assertTrue(np.isfinite(value[0]))
        with self.assertRaises(ValueError): PCEncoder(NeverPredict(),ToyCoder()).encode(value)

    def test_scalar_partial_block_and_bad_shapes(self):
        x=np.array([1,2,3]); a=PCEncoder(StatefulPredictor(),ToyCoder(),n_predictions=2); b=PCDecoder(StatefulPredictor(),ToyCoder(),n_predictions=2)
        self.assertEqual(b.decode(a.encode(x)),list(x))
        with self.assertRaises(ValueError): a.encode([[1,2]])
        with self.assertRaises(ValueError): PCEncoder(StatefulPredictor(),ToyCoder(),n_predictions=0)
    def test_multivariate_d_by_n_protocol_roundtrip(self):
        x=np.array([[1,2],[3,4],[5,6]])
        enc=MultivariatePCEncoder(MatrixPredictor(),[ToyCoder(),ToyCoder()],[UniformQuantizer(1),UniformQuantizer(1)],2)
        dec=MultivariatePCDecoder(MatrixPredictor(),[ToyCoder(),ToyCoder()],[UniformQuantizer(1),UniformQuantizer(1)],2)
        streams=enc.encode(x); np.testing.assert_array_equal(dec.decode(streams),x)
        with self.assertRaises(ValueError): enc.encode(x.T)
        with self.assertRaises(ValueError): dec.decode([streams[0]])
    def test_stateful_predictor_windows_match_across_blocks(self):
        x=np.array([1,2,3,4,5])
        ep,dp=StatefulPredictor(),StatefulPredictor()
        enc=PCEncoder(ep,ToyCoder(),UniformQuantizer(1),n_predictions=2)
        dec=PCDecoder(dp,ToyCoder(),UniformQuantizer(1),n_predictions=2)
        np.testing.assert_allclose(dec.decode(enc.encode(x)),x)
        np.testing.assert_array_equal(ep.window,dp.window)
        x2=np.array([[1,4],[2,5],[3,6],[4,7],[5,8]])
        ep,dp=MatrixPredictor(),MatrixPredictor()
        qs=[UniformQuantizer(1),UniformQuantizer(1)]
        enc=MultivariatePCEncoder(ep,[ToyCoder(),ToyCoder()],qs,2)
        dec=MultivariatePCDecoder(dp,[ToyCoder(),ToyCoder()],qs,2)
        np.testing.assert_allclose(dec.decode(enc.encode(x2)),x2)
        np.testing.assert_array_equal(ep.windows,dp.windows)

    def test_multivariate_bad_streams_and_prediction_orientation(self):
        class WrongOrientation:
            def predict(self,n): return np.zeros((n,2))
            def update(self,x): raise AssertionError('update called')
        dec=MultivariatePCDecoder(WrongOrientation(),[ToyCoder(),ToyCoder()],[UniformQuantizer(1),UniformQuantizer(1)])
        for streams in ([b'[]',b'[]'], [b'',b'[]'], [b'[]',object()]):
            with self.subTest(streams=streams), self.assertRaises(ValueError): dec.decode(streams)
        with self.assertRaises(ValueError): dec.decode([b'[1]',b'[1]'])

    def test_multivariate_unequal_channel_lengths_and_quantizer_modes(self):
        class Unequal(ToyCoder):
            def decode_symbols(self, b): return [1] if b == b'a' else [1,2]
        dec=MultivariatePCDecoder(MatrixPredictor(),[Unequal(),Unequal()],[UniformQuantizer(1),UniformQuantizer(1)])
        with self.assertRaises(ValueError): dec.decode([b'a',b'b'])
        clipped=UniformQuantizer(1,levels=2,clip=True)
        self.assertEqual(clipped.value_to_symbol(100),1)
        unclipped=UniformQuantizer(1,levels=2,clip=False)
        with self.assertRaises(ValueError): unclipped.value_to_symbol(100)

    def test_ans_optional_quantizer_same_format(self):
        ft=SimpleFrequencyTable([1,1,1,1]); symbols=[0,1,2,3,1]
        a=ANSCoder(ft); b=ANSCoder(ft,UniformQuantizer(1))
        self.assertEqual(a.encode_symbols(symbols),b.encode_symbols(symbols))
        self.assertEqual(a.decode_symbols(a.encode_symbols(symbols)),symbols)

if __name__ == '__main__': unittest.main()
