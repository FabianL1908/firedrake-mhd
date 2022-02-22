import argparse
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--Res", nargs='+', type=int)
parser.add_argument("--Rems", nargs='+', type=int)
parser.add_argument("--S", nargs='+', type=int)
parser.add_argument("--testproblem", type=str, required=True)

args, _ = parser.parse_known_args()
Rems = args.Rems
Res = args.Res
S = args.S
testproblem = args.testproblem

if len(args.S)!=1 and len(args.Rems)!=1:
    S = args.Res
    Res = args.S

for linearisation in ["picard", "mdp", "newton"]:
    res_list = []
    for rem in Rems:
        for s in S:
            for re in Res:
                try:
                    with open('results/results'+str(linearisation)+str(testproblem)+'/'+str(float(re))+str(float(rem*s))+'.txt','r') as f:
                            res_list.append(f.read())
                except:
                    res_list.append("    -   ")
    temp = sys.stdout           
    f = open('output_'+str(testproblem)+'.txt','a')
    sys.stdout = f
    print(linearisation)
    if len(S)!=1:
        print("  S\Re   ", end = '')
        iterRems = iter(Rems[0]*S)
    elif len(Rems)!=1:
        if len(args.Res)!=1:
            print("  Rem\Re   ", end = '')
        else:
            print("  Rem\S   ", end = '')
        iterRems = iter(Rems*S[0])
    for re in Res:
            print("%8s &" % re, end = ' ')
    for i, result in enumerate(res_list):
            if (i) % len(Res) == 0:
               print(" ")
               print("%8s &" % next(iterRems) , end = ' ')
            print(result + " &" , end = ' ')
    print(" "); print("")

    sys.stdout = temp
    f.close()
    with open('output_'+str(testproblem)+'.txt','r') as f:
       print(f.read())
