package lpsolver;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import edu.harvard.econcs.jopt.solver.IMIPResult;
import edu.harvard.econcs.jopt.solver.IMIPSolver;
import edu.harvard.econcs.jopt.solver.SolveParam;
import edu.harvard.econcs.jopt.solver.client.SolverClient;
import edu.harvard.econcs.jopt.solver.mip.Constraint;
import edu.harvard.econcs.jopt.solver.mip.MIPWrapper;
import edu.harvard.econcs.jopt.solver.mip.Variable;
public class LpSolver {

	public static void doWork(){

        MIPWrapper mipWrapper = MIPWrapper.makeNewMaxMIP();

        Variable x = mipWrapper.makeNewDoubleVar("x");
        x.setLowerBound(0);
        Variable y = mipWrapper.makeNewDoubleVar("y");
        y.setLowerBound(0);
        Variable z = mipWrapper.makeNewDoubleVar("z");
        z.setLowerBound(0);

        mipWrapper.addObjectiveTerm(15, x);
        mipWrapper.addObjectiveTerm(20, y);
        mipWrapper.addObjectiveTerm(25, z);

        Constraint mashineA = mipWrapper.beginNewLEQConstraint(42);
        mashineA.addTerm(3, x);
        mashineA.addTerm(2, y);
        mashineA.addTerm(3, z);
        mipWrapper.endConstraint(mashineA);

        Constraint mashineB = mipWrapper.beginNewLEQConstraint(36);
        mashineB.addTerm(2, x);
        mashineB.addTerm(3, y);
        mashineB.addTerm(2, z);
        mipWrapper.endConstraint(mashineB);

        Constraint mashineC = mipWrapper.beginNewLEQConstraint(48);
        mashineC.addTerm(6, x);
        mashineC.addTerm(3, y);
        mashineC.addTerm(4, z);
        mipWrapper.endConstraint(mashineC);
       
        Constraint mashineD = mipWrapper.beginNewLEQConstraint(50);
        mashineD.addTerm(1, x);
        mipWrapper.endConstraint(mashineD);
        
        Constraint mashineE = mipWrapper.beginNewLEQConstraint(40);
        mashineE.addTerm(1, y);
        mipWrapper.endConstraint(mashineE);
        
        Constraint mashineF = mipWrapper.beginNewLEQConstraint(30);
        mashineF.addTerm(1, z);
        mipWrapper.endConstraint(mashineF);
        
        Constraint mashineG = mipWrapper.beginNewGEQConstraint(0);
        mashineG.addTerm(-0.6, x);
        mashineG.addTerm(0.4, y);
        mashineG.addTerm(0.4, z);
        mipWrapper.endConstraint(mashineG);
        
        Constraint mashineH = mipWrapper.beginNewGEQConstraint(0);
        mashineH.addTerm(0.4, x);
        mashineH.addTerm(-0.6, y);
        mashineH.addTerm(0.4, z);
        mipWrapper.endConstraint(mashineH);
       
        Constraint mashineI = mipWrapper.beginNewGEQConstraint(0);
        mashineI.addTerm(0.4, x);
        mashineI.addTerm(0.4, y);
        mashineI.addTerm(-0.6, z);
        mipWrapper.endConstraint(mashineI);
        IMIPSolver solver = new SolverClient();
        IMIPResult result = solver.solve(mipWrapper);  
        System.out.println("Results:");
        // Get specific var value
        System.out.println("Var X "+ result.getValue("x"));
        System.out.println("---------------------");
        // Print full program
        System.out.println(result.toString(mipWrapper));
	}
}
