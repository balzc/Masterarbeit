package simulation;

import org.apache.commons.math3.distribution.NormalDistribution;

public class Simulation {
	public void work(){
		/* create User
		 * get User feedback
		 * learn Parameters
		 * make Priceprediction
		 * Solve MDP
		 * Load according to MDP
		 * calculate Utility
		 */
		Double vmintrue = 10.;
		Double qmintrue = 10.;
		Double tstarttrue = 90.;
		Double tcrittrue = 95.;
		Double mqtrue = 2.;
		Double sd = 0.05;
		int counter = 0;
		int numberOfRuns = 2;
		double totalUtility = 0;
		while(counter <= numberOfRuns){

		

			Double vmin = sampleFromNormal(vmintrue, sd);
			Double qmin = sampleFromNormal(qmintrue, sd);
			Double tstart = sampleFromNormal(tstarttrue, sd);
			Double tcrit = sampleFromNormal(tcrittrue, sd);
			Double mq = sampleFromNormal(mqtrue, sd);
			
			
			counter++;
		}
	}
	
	public double sampleFromNormal(double mean, double sd){
		NormalDistribution nd = new NormalDistribution(mean, sd);
		return nd.sample();
	}
	
}
