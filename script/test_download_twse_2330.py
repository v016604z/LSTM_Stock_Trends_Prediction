import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import json

P = Path(__file__).with_name('download_twse_2330.py')
spec = importlib.util.spec_from_file_location('twse_downloader', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def payload(month='20190101'):
    return {
        'stat':'OK',
        'date':month,
        'fields':['日期','成交股數','成交金額','開盤價','最高價','最低價','收盤價','漲跌價差','成交筆數','註記'],
        'data': [['108/01/02','10,000,000','2,650,000,000','265.50','266.00','263.50','265.00','+1.00','5,000',''],
                 ['108/01/03','7,000,001','2,500,000,000','260.00','263.50','250.00','255.50','-9.50','6,000','']]
    }


class Tests(unittest.TestCase):
    def test_parse_realistic_month(self):
        out=m.parse_month(payload(), '20190101')
        self.assertEqual(len(out),2)
        self.assertEqual(out[0], {'Date': '2019-01-02', 'Open':'265.50','High':'266.00','Low':'263.50','Close':'265.00','Volume':'10000000'})
        self.assertEqual(out[1]['Volume'],'7000001')

    def test_validation(self):
        for kind in ('bad_stat','wrong_month','missing_column','zero_volume','nan','out_of_order'):
            with self.subTest(kind=kind):
                p=payload()
                if kind=='bad_stat':p['stat']='查無資料'
                if kind=='wrong_month':p['data'][0][0]='108/02/01'
                if kind=='missing_column':p['fields'][1]='other'
                if kind=='zero_volume':p['data'][1][1]='0'
                if kind=='nan':p['data'][1][3]='--'
                if kind=='out_of_order':p['data'].reverse()
                with self.assertRaises(m.DownloadError):m.parse_month(p,'20190101')

    def test_month_generator(self):
        self.assertEqual(list(m.months('2025-11','2026-02')), ['20251101','20251201','20260101','20260201'])
        self.assertEqual(len(list(m.months('2015-01','2025-12'))),132)

    def test_output_metadata_and_no_overwrite(self):
        with TemporaryDirectory() as p:
            d=Path(p)
            rows=m.parse_month(payload(),'20190101')
            paths=m.write_outputs(d,'2330','2019-01','2019-01',rows,[],[x['Date'] for x in rows],{'name':'XTAI','source':'test'})
            self.assertEqual(len(paths),3)
            self.assertIn(b'Date,Open,High,Low,Close,Volume',paths[0].read_bytes())
            metadata=json.loads(paths[2].read_text())
            self.assertEqual(metadata['source']['name'],'TWSE_STOCK_DAY')
            self.assertEqual(metadata['source']['price_basis'],'unadjusted')
            self.assertEqual(metadata['data']['row_count'],2)
            with self.assertRaises(m.DownloadError):
                m.write_outputs(d,'2330','2019-01','2019-01',rows,[],[x['Date'] for x in rows],{})


if __name__=='__main__':unittest.main(verbosity=2)
