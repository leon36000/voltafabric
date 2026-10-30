import pathlib,sys,unittest,torch
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));from verify import accumulate
class TestReference(unittest.TestCase):
 def test_weighted_routes_once_and_permutation(self):
  routes=[[{'expert':2,'weight':.25},{'expert':0,'weight':.5}],[{'expert':1,'weight':.125},{'expert':2,'weight':.75}]];rows=[torch.tensor([1.]),torch.tensor([2.])]
  def forward(expert,row):return torch.tensor([float(row[0])*(expert+1),10.0+expert],dtype=torch.float16)
  out=accumulate(routes,rows,forward);self.assertTrue(torch.equal(out,torch.tensor([[1.25,8.0],[5.0,10.375]],dtype=torch.float64)))
 def test_duplicate_and_nonfinite_refused(self):
  for routes in [[{'expert':0,'weight':.5},{'expert':0,'weight':.5}],[{'expert':1,'weight':float('nan')}]]:
   with self.assertRaises(ValueError):accumulate([routes],[torch.zeros(1)],lambda e,x:torch.zeros(1))
if __name__=='__main__':unittest.main()
