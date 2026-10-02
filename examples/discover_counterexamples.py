"""Run declared property probes and preserve semantic counterexamples."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qvm import QVM,default_property_report
ROOT=Path(__file__).resolve().parents[1]
def main():
 report=default_property_report(QVM());out=ROOT/'artifacts/property_report.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps({'properties':len(report['results']),'counterexamples':len(report['counterexamples']),'output':str(out)},indent=2))
if __name__=='__main__':main()
