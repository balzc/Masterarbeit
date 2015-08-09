package simulation;

import gp.GP;

import java.util.ArrayList;

import learning.BayesInf;
import main.Main;
import mdp.EVMDP;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.jblas.DoubleMatrix;
import org.jblas.util.Random;

import agents.LowPriceLoader;
import agents.PlugAndForgetLoader;
import agents.SortedMinLoader;
import util.FileHandler;
import cov.Additive;
import cov.CovarianceFunction;
import cov.Matern;
import cov.Multiplicative;
import cov.OrnsteinUhlenbeck;
import cov.Periodic;
import cov.SquaredExponential;

public class Simulation {

	/*
	 * @param blablabla
	 */
	public void work(String fhprices, String fhout, double vminvarInput, double qminInput, double qmaxInput, double tstartInput, double tcritInput, double mqInput, double kwhPerUnitInput,double bisd){
		/* create User
		 * get User feedback
		 * learn Parameters
		 * make Priceprediction
		 * Solve MDP
		 * Load according to MDP
		 * calculate Utility
		 */
		boolean PROFILING = false;
		long time = (long)0.;

		// user intrinsic parameters
		// input is the kwh of a car battery, qmax is then the number of timesteps required to fill that battery with kwhPerUnitInput
		double qmax = qmaxInput/kwhPerUnitInput;
		// input is a percentage number, qminTrue is then the number of timesteps required to fill the battery to that percentage
		double qminTrue = qminInput*qmaxInput/kwhPerUnitInput;
		// input is a price per kwh, vmintrue is then the number of units required for qmin times the price per kwh times kwh per unit
		double vminTrue = vminvarInput*qminTrue*kwhPerUnitInput;
		// input relative to time discretization
		double tstartTrue = tstartInput;
		// input relative to tstart, tcritTrue value relative to time discretization, tcritTrue is no offset anymore
		double tcritTrue = tcritInput + tstartInput;
		// input is price per kwh, mqTrue is then price per loadunit
		double mqTrue = mqInput*kwhPerUnitInput;

		boolean log = false;
		String runId = "qmax " + qmax + "  qmin " + qminTrue + " vmin " + vminTrue + " mq " + mqTrue + " kwh " + kwhPerUnitInput + " bisd "+ bisd;
		System.out.println("qmax " + qmax + "  qmin " + qminTrue + " vmin " + vminTrue + " mq " + mqTrue + " kwh " + kwhPerUnitInput + " bisd "+ bisd);
		double bayesInfSD = bisd;
		double vminSD = 1.;// mqsd *qmin
		double mqSD = 1.;//5
		double tdepTrueSD = 0.05;//15 min
		double bayesInfVar = bayesInfSD*bayesInfSD;

		double tdepTrueMean = tstartTrue;
		double tdepLearnedMean = tstartTrue;
		double tdepLearnedSD = tdepTrueSD;
		double returnLoadSD = 2;
		double vmin = sampleFromNormal(vminTrue, vminSD);
		double mq = sampleFromNormal(mqTrue, mqSD);
		double endOfDayOffset = 0;
		double kwhPerUnit = kwhPerUnitInput;

		double currentLoadMDP = 0;
		double currentLoadLPL = 0;
		double currentLoadPAFL = 0;
		double currentLoadSML = 0;

		// Stopping criterion parameters
		int numberOfSamplesForStoppingCriterion = 1000;
		int numberOfConcurrentThreads = 100;

		double stoppingCriterionThreshold = 0.5;

		// general simulation parameters
		ArrayList<Double> tdepMeans = new ArrayList<Double>();
		ArrayList<Double> vminList = new ArrayList<Double>();
		ArrayList<Double> mqList = new ArrayList<Double>();
		String fileHandleOut = fhout;
		double priceOffset = 0;
		int counter = 0;
		int numberOfRuns = 365;
		double totalUtilityMDP = 0;
		double totalUtilityLPL = 0;
		double totalUtilityPAFL = 0;
		double totalUtilitySML = 0;

		ArrayList<Double> additionalLoadDataQ = new ArrayList<Double>();
		ArrayList<Double> additionalLoadDataV = new ArrayList<Double>();
		double[] loadDataQuestions = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1};
		boolean stopAsking = false;
		// Price Prediction parameters
		String fileHandlePrices = fhprices;//"/users/balz/documents/workspace/masterarbeit/data/prices2.csv";
		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);

		double[] priceParameters = {1.0368523093853659,5.9031209989290048,1.0368523093853659,.3466674176616187,3.5551018122094575,8.1097474657929007};
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		OrnsteinUhlenbeck ou = new OrnsteinUhlenbeck();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, ou);
		CovarianceFunction cf = a1;
		double gpVar = 0.5;
		DoubleMatrix parameters = new DoubleMatrix(priceParameters);
		int learnStart = 0;
		int learnSize = 96;
		int predictStart = learnStart + learnSize;
		int predictSize = 96;
		int dataPointsPerDay = 96;
		double treturnMean = 0;
		double treturnSD = 0.05;
		double unpluggedDuration = 28;
		double stepSize = 1./dataPointsPerDay;

		// MDP Parameters
		double deltaPrice = 1;
		int numSteps = 96;

		// 
		LowPriceLoader LPL = new LowPriceLoader(mq, vmin, qminTrue, qmax);
		PlugAndForgetLoader PAFL = new PlugAndForgetLoader(qmax);
		ArrayList<Double> loadsMDP = new ArrayList<Double>();
		ArrayList<Double> loadsLPL = new ArrayList<Double>();
		ArrayList<Double> loadsPAFL = new ArrayList<Double>();
		ArrayList<Double> loadsSML = new ArrayList<Double>();

		ArrayList<Double> loadsDailyMDP = new ArrayList<Double>();
		ArrayList<Double> loadsDailyLPL = new ArrayList<Double>();
		ArrayList<Double> loadsDailyPAFL = new ArrayList<Double>();
		ArrayList<Double> loadsDailySML = new ArrayList<Double>();

		ArrayList<Double> realPrices = new ArrayList<Double>();
		ArrayList<Double> predictedPrices = new ArrayList<Double>();

		ArrayList<Double> actionsMDP = new ArrayList<Double>();
		ArrayList<Double> actionsLPL = new ArrayList<Double>();
		ArrayList<Double> actionsPAFL = new ArrayList<Double>();
		ArrayList<Double> actionsSML = new ArrayList<Double>();

		ArrayList<Double> costsDailyMDP = new ArrayList<Double>();
		ArrayList<Double> costsDailyLPL = new ArrayList<Double>();
		ArrayList<Double> costsDailyPAFL = new ArrayList<Double>();
		ArrayList<Double> costsDailySML = new ArrayList<Double>();

		ArrayList<Double> costsPerDeltaTMDP = new ArrayList<Double>();
		ArrayList<Double> costsPerDeltaTLPL = new ArrayList<Double>();
		ArrayList<Double> costsPerDeltaTPAFL = new ArrayList<Double>();
		ArrayList<Double> costsPerDeltaTSML = new ArrayList<Double>();

		ArrayList<Double> totalUtilitiesMDP = new ArrayList<Double>();
		ArrayList<Double> totalUtilitiesLPL = new ArrayList<Double>();
		ArrayList<Double> totalUtilitiesPAFL = new ArrayList<Double>();
		ArrayList<Double> totalUtilitiesSML = new ArrayList<Double>();

		ArrayList<Double> regret = new ArrayList<Double>();
		ArrayList<Double> mqDiffs = new ArrayList<Double>();
		ArrayList<Double> vminDiffs = new ArrayList<Double>();

		ArrayList<Double> timeStepsCounter = new ArrayList<Double>();

		// Start simulation
		while(counter < numberOfRuns){
			System.out.println("Run " + counter + "   " + runId);
			// sample the observed values from normal distributions
			//!!! values need to be consistent
			// qmin, tstart & tcrit actual values
			if(PROFILING){
				time = System.nanoTime();
			}
			double qmin = qminTrue;//sampleFromNormal(qmintrue, sd);
			double tstart = tstartTrue;//sampleFromNormal(tstarttrue, sd);
			double tcrit = tcritTrue;//sampleFromNormal(tcrittrue, sd);
			// if we don't have reached the stopping criterion yet ask the user about mq and vmin
			if(!stopAsking){
				mq = sampleFromNormal(mqTrue, mqSD);
				while(mq < 0 ){
					mq = sampleFromNormal(mqTrue, mqSD);
				}
				mqList.add(mq);
				vmin = sampleFromNormal(vminTrue, vminSD);
				while(vmin < 0){
					vmin = sampleFromNormal(vminTrue, vminSD);
				}
				vminList.add(vmin);
			}
			double tdep = sampleFromNormal(tdepTrueMean, tdepTrueSD);
			while(!isInRange(tdep, tstart, tcrit-tstart)){
				tdep = sampleFromNormal(tdepTrueMean, tdepTrueSD);
			}
			// update the learned mean departure time with new information
			tdepMeans.add(tdep);
			double tdepSum = 0;
			for(Double d: tdepMeans){
				tdepSum += d;
			}
			double tdepSampleMean = tdepSum/tdepMeans.size();
			double tdepVar = tdepLearnedSD*tdepLearnedSD;
			tdepLearnedMean = (tdepSampleMean*tdepVar/(tdepMeans.size()*tdepVar+tdepVar))+(tdepMeans.size()*tdepLearnedMean*tdepVar/(tdepMeans.size()*tdepVar+tdepVar));

			// choose a Question at random and add the additional data
			int rndQuestion = Random.nextInt(loadDataQuestions.length);
			additionalLoadDataQ.add(qmin+(qmax-qmin)*loadDataQuestions[rndQuestion]);
			additionalLoadDataV.add((vmin-qmin*mq)+(qmin + (qmax-qmin)*loadDataQuestions[rndQuestion])*mq);
			DoubleMatrix xLoad = DoubleMatrix.concatVertically((DoubleMatrix.ones(counter+1)).transpose(),(new DoubleMatrix(additionalLoadDataQ)).transpose());
			if(PROFILING){
				System.out.println("Init Data - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			if(PROFILING){
				time = System.nanoTime();
			}
			// do Bayesian inference to get mq and vmin
			BayesInf bi = new BayesInf(xLoad, new DoubleMatrix(additionalLoadDataV), bayesInfVar);
			//			Main.printMatrix(xLoad);
			//			Main.printMatrix(new DoubleMatrix(additionalLoadDataV));
			bi.setup();
			if(PROFILING){
				System.out.println("Bayesian Inference - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			double mqLearned = bi.getMean().get(1);
			double vminLearned =  bi.getMean().get(0)+mqLearned*qmin;
			if(log)
			System.out.println("qmin "+ qmin);

			if(PROFILING){
				time = System.nanoTime();
			}
			//prepare GP
			double[] xTrain = new double[learnSize];
			for(int o = 0; o < learnSize; o++){
				xTrain[o] = o*stepSize+learnStart*stepSize;
			}
			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
			double[] xTest = new double[predictSize];
			for(int o = 0; o < predictSize; o++){
				xTest[o] = o*stepSize + predictStart*stepSize;
			}
			DoubleMatrix xTestM = new DoubleMatrix(xTest);
			//			System.out.println("Ahi" + priceSamples.rows);
			DoubleMatrix yTrainMPrices = Main.subVector(learnStart, learnStart+learnSize, priceSamples);
			DoubleMatrix realPriceMatrix = Main.subVector(predictStart, predictStart+predictSize, priceSamples);
			GP priceGP = new GP(xTrainM,xTestM,parameters,cf,gpVar);
			//			Main.printMatrix(xTestM);
			//			Main.printMatrix(xTrainM);
			//						Main.printMatrix(yTrainMPrices);
			priceGP.setup(yTrainMPrices);
			if(PROFILING){
				System.out.println("Gaussian Process - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			DoubleMatrix predMeanPrices = priceGP.getPredMean().add(priceOffset);
			DoubleMatrix predVarPrices = priceGP.getPredVar();
			//			Main.printMatrix(predMeanPrices);
			// setup mdp
			if(PROFILING){
				time = System.nanoTime();
			}
			if(log){
				System.out.println("tdep: " + (tdepLearnedMean  + endOfDayOffset) );
				System.out.println("tstart: " + (tstart  + endOfDayOffset) );

				System.out.println("tcrit: " + (tcrit  + endOfDayOffset) );
				System.out.println("vmin: " + vmin  );
				System.out.println("mq: " + mq  );

			}
			EVMDP mdp = new EVMDP(predMeanPrices, predVarPrices, deltaPrice, numSteps, qmax, qmin, mqLearned, tstart + endOfDayOffset, tcrit + endOfDayOffset, vminLearned,tdepLearnedMean  + endOfDayOffset,tdepLearnedSD,currentLoadMDP,kwhPerUnit);
			mdp.setup();
			// charge the vehicle according to policy and update total utility
			if(PROFILING){
				System.out.println("mdp calculation - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			// sample tdep and load till tdep, calculate utility until tdep
			if(PROFILING){
				time = System.nanoTime();
			}
			// Check stopping criterion
			if(!stopAsking){
				int stoppingRunsCounter = 0;
				double expectedUtility = mdp.getqValues()[0][mdp.priceToState(realPriceMatrix.get(0)+priceOffset)][mdp.loadToState(currentLoadMDP)][0];
				double cumulatedDifference = 0;
				double[][] covarianceMatrix = new double[2][2];
				covarianceMatrix[0] = new double[] {bi.getAInv().get(0,0),bi.getAInv().get(0,1)};
				covarianceMatrix[1] = new double[] {bi.getAInv().get(1,0),bi.getAInv().get(1,1)};
				MultivariateNormalDistribution mvnd = new MultivariateNormalDistribution(new double[] {bi.getMean().get(0),bi.getMean().get(1)},covarianceMatrix);
				for(int o = 0; o < numberOfSamplesForStoppingCriterion/numberOfConcurrentThreads; o++){
					if(log)
					System.out.println("stoppinval run: "+ o + " " + numberOfSamplesForStoppingCriterion/numberOfConcurrentThreads + "    " + runId);
					SimulationThread[] sts = new SimulationThread[numberOfConcurrentThreads];
					// start threads
					for(int i = 0; i < numberOfConcurrentThreads; i++){
						double[] sample = mvnd.sample();
						double sampledVmin = sample[0] + qmin * sample[1];
						double sampledMQ = sample[1];
						if(log)
							System.out.println(i +  " " + mqLearned + " " + vminLearned + " " + sampledVmin + " vmin mq " + sampledMQ + " eu " + expectedUtility + " sample 0 " + sample[0] + " sample 1 " + sample[1] + " bi 0 " + bi.getMean().get(0)+ " bi 1 " + bi.getMean().get(1));
						EVMDP samplemdp = new EVMDP(sampledMQ, sampledVmin, kwhPerUnit, mdp.getPriceProb(), mdp.getLoadProb(), mdp.getEndStateProb(), mdp.getPrices(), mdp.getLoads(), numSteps,tstart + endOfDayOffset, tcrit + endOfDayOffset,tdepLearnedMean  + endOfDayOffset, qmin, qmax);


						int[] indexes = new int[2];
						indexes[0] = mdp.priceToState(realPriceMatrix.get(0)+priceOffset);
						indexes[1] = mdp.loadToState(currentLoadMDP);
						SimulationThread thread = new SimulationThread(samplemdp,stoppingRunsCounter,cumulatedDifference,expectedUtility,indexes);
						thread.start();
						sts[i] = thread;

					}
					// collect threads and sum cumulateddifferences
					for(int i = 0; i < numberOfConcurrentThreads; i++){
						SimulationThread b = sts[i];
						try {
							b.thread.join();
							if(log)
							System.out.println("cd " + cumulatedDifference);
							cumulatedDifference += b.value;

						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							System.out.println(e.getMessage());
							e.printStackTrace();
						}

					}
				}
				double avgRegret = Math.abs(cumulatedDifference/numberOfSamplesForStoppingCriterion);
				if(log)
					System.out.println("stoppinval " + avgRegret);
				if(avgRegret < stoppingCriterionThreshold){
					stopAsking = true;
				}
				regret.add(avgRegret);
				mqDiffs.add(mq-mqLearned);
				vminDiffs.add(vminTrue-vminLearned);
			}
			if(PROFILING){
				System.out.println("stopping criterion - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			// intialise the sorted min loader
			SortedMinLoader sml = new SortedMinLoader();
			sml.setup(Main.subVector(0, (int)(tdep + endOfDayOffset+1), predMeanPrices), qmax, currentLoadSML/kwhPerUnit);
			// load according to the different agents
			for(int o = 0; o < tdep + endOfDayOffset; o++){
				if(mdp.priceToState(realPriceMatrix.get(o)+priceOffset) < 0){
					System.out.println("Bad Price" + realPriceMatrix.get(o)+priceOffset);
				}
				if(mdp.loadToState(currentLoadMDP) < 0){
					System.out.println("Bad Load" + currentLoadMDP);
				}

				int actionMDP = mdp.getOptPolicy()[o][mdp.priceToState(realPriceMatrix.get(o)+priceOffset)][mdp.loadToState(currentLoadMDP)][0];
				// MDP: cost summation, save for every day			
				totalUtilityMDP += mdp.rewards(currentLoadMDP, actionMDP, realPriceMatrix.get(o)+priceOffset,o,0);
				costsPerDeltaTMDP.add(mdp.rewards(currentLoadMDP, actionMDP, realPriceMatrix.get(o)+priceOffset,o,0));
				currentLoadMDP = mdp.updateLoad(currentLoadMDP, actionMDP);
				loadsMDP.add(currentLoadMDP);
				actionsMDP.add((double)actionMDP);
				// Low price loader
				int actionLPL = LPL.actionQuery(currentLoadLPL/kwhPerUnit, realPriceMatrix.get(o)+priceOffset);
				totalUtilityLPL += mdp.rewards(currentLoadLPL, actionLPL, realPriceMatrix.get(o)+priceOffset,o,0);
				costsPerDeltaTLPL.add(mdp.rewards(currentLoadLPL, actionLPL, realPriceMatrix.get(o)+priceOffset,o,0));
				currentLoadLPL = mdp.updateLoad(currentLoadLPL, actionLPL);
				actionsLPL.add((double)actionLPL);
				loadsLPL.add(currentLoadLPL);
				// plug and forget loader
				int actionPAFL = PAFL.actionQuery(currentLoadPAFL/kwhPerUnit);
				totalUtilityPAFL += mdp.rewards(currentLoadPAFL, actionPAFL, realPriceMatrix.get(o)+priceOffset,o,0);
				costsPerDeltaTPAFL.add(mdp.rewards(currentLoadPAFL, actionPAFL, realPriceMatrix.get(o)+priceOffset,o,0));
				currentLoadPAFL = mdp.updateLoad(currentLoadPAFL, actionPAFL);
				actionsPAFL.add((double)actionPAFL);
				loadsPAFL.add(currentLoadPAFL);
				// SortedMinLoader
				int actionSML = sml.policy[o];
				totalUtilitySML += mdp.rewards(currentLoadSML, actionSML, realPriceMatrix.get(o)+priceOffset,o,0);
				costsPerDeltaTSML.add(mdp.rewards(currentLoadSML, actionSML, realPriceMatrix.get(o)+priceOffset,o,0));
				currentLoadSML = mdp.updateLoad(currentLoadSML, actionSML);
				actionsSML.add((double)actionSML);
				loadsSML.add(currentLoadSML);
				//record prices
				realPrices.add(realPriceMatrix.get(o));
				predictedPrices.add(predMeanPrices.get(o));
				timeStepsCounter.add((double)o);
			}
			if(log){
				System.out.println("endload MDP: " + currentLoadMDP);
				System.out.println("endload LPL: " + currentLoadLPL);
				System.out.println("endload PAFL: " + currentLoadPAFL);
				System.out.println("endload SML: " + currentLoadSML);

				System.out.println("costs MDP: " + totalUtilityMDP);
				System.out.println("costs LPL: " + totalUtilityLPL);
				System.out.println("costs PAFL: " + totalUtilityPAFL);
				System.out.println("costs SML: " + totalUtilitySML);
			}
			loadsDailyLPL.add(currentLoadLPL);
			loadsDailyMDP.add(currentLoadMDP);
			loadsDailyPAFL.add(currentLoadPAFL);
			loadsDailySML.add(currentLoadSML);
			costsDailyLPL.add(totalUtilityLPL);
			costsDailyMDP.add(totalUtilityMDP);
			costsDailyPAFL.add(totalUtilityPAFL);
			costsDailySML.add(totalUtilitySML);
			totalUtilityMDP = totalUtilityMDP + mdp.rewards(currentLoadMDP, 0, 0, (int)tdep, 1);
			totalUtilityLPL = totalUtilityLPL + mdp.rewards(currentLoadLPL, 0, 0, (int)tdep, 1);
			totalUtilityPAFL = totalUtilityPAFL + mdp.rewards(currentLoadPAFL, 0, 0, (int)tdep, 1);
			totalUtilitySML = totalUtilitySML + mdp.rewards(currentLoadSML, 0, 0, (int)tdep, 1);
			if(log){
				System.out.println("total utility MDP: " + totalUtilityMDP);
				System.out.println("total utility LPL: " + totalUtilityLPL);
				System.out.println("total utility PAFL: " + totalUtilityPAFL);
				System.out.println("total utility SML: " + totalUtilitySML);
			}
			counter++;
			double usedLoad = currentLoadMDP;
			if(currentLoadMDP > 0){
				double tempload = qmin*kwhPerUnit;
				usedLoad = (int)sampleFromNormal(tempload, returnLoadSD);
				while(usedLoad > currentLoadMDP || usedLoad < tempload){
					usedLoad = (int)sampleFromNormal(tempload, returnLoadSD);
				}
				if(log)
					System.out.println("used Load " + usedLoad);
				currentLoadMDP = currentLoadMDP - usedLoad;
				if(log)
					System.out.println("used Load " + usedLoad);
			}
			currentLoadLPL = Math.max(0,currentLoadLPL - usedLoad);
			currentLoadPAFL = Math.max(0,currentLoadPAFL - usedLoad);
			currentLoadSML = Math.max(0,currentLoadSML - usedLoad);

			if(log){
				System.out.println("returnload MDP: " + currentLoadMDP);
				System.out.println("returnload LPL: " + currentLoadLPL);
				System.out.println("returnload PAFL: " + currentLoadPAFL);
				System.out.println("returnload SML: " + currentLoadSML);

				// TODO:
				System.out.println("predicstart: " + predictStart);
			}
			treturnMean = sampleFromNormal(predictStart + tdep + endOfDayOffset + unpluggedDuration, treturnSD);
			learnStart = (int)(treturnMean - learnSize);
			predictStart = (int)treturnMean;
			endOfDayOffset = numSteps-treturnMean%numSteps;
			if(log){
				System.out.println("tret: " + treturnMean);
				System.out.println("endofday offset: " + endOfDayOffset);
				System.out.println();
			}
			totalUtilitiesLPL.add(totalUtilityLPL);
			totalUtilitiesMDP.add(totalUtilityMDP);
			totalUtilitiesSML.add(totalUtilitySML);
			totalUtilitiesPAFL.add(totalUtilityPAFL);

			totalUtilityMDP = 0;
			totalUtilityLPL = 0;
			totalUtilityPAFL = 0;
			totalUtilitySML = 0;



		}
		//		printOut = DoubleMatrix.concatHorizontally(DoubleMatrix.concatHorizontally(DoubleMatrix.concatHorizontally(new DoubleMatrix(loads), new DoubleMatrix(realPrices)),new DoubleMatrix(predictedPrices)),new DoubleMatrix(totalUtilities));
		// Simulation ends and we safe the data
		//		Main.printMatrix(new DoubleMatrix(loadsMDP));
		//		Main.printMatrix(new DoubleMatrix(loadsLPL));
		//		Main.printMatrix(new DoubleMatrix(loadsPAFL));
		//		Main.printMatrix(new DoubleMatrix(loadsSML));
		//
		//		Main.printMatrix(new DoubleMatrix(predictedPrices));
		//		Main.printMatrix(new DoubleMatrix(realPrices));
		DoubleMatrix dailyReport = DoubleMatrix.concatHorizontally(new DoubleMatrix(totalUtilitiesMDP), new DoubleMatrix(totalUtilitiesLPL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(totalUtilitiesPAFL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(totalUtilitiesSML));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(costsDailyMDP));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(costsDailyLPL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(costsDailyPAFL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(costsDailySML));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(loadsDailyMDP));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(loadsDailyLPL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(loadsDailyPAFL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(loadsDailySML));

		DoubleMatrix timeStepReport = DoubleMatrix.concatHorizontally(new DoubleMatrix(actionsMDP), new DoubleMatrix(actionsLPL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(actionsPAFL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(actionsSML));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(loadsMDP));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(loadsLPL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(loadsPAFL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(loadsSML));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(costsPerDeltaTMDP));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(costsPerDeltaTLPL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(costsPerDeltaTPAFL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(costsPerDeltaTSML));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(predictedPrices));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(realPrices));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(timeStepsCounter));

		DoubleMatrix stoppingReport = DoubleMatrix.concatHorizontally(DoubleMatrix.concatHorizontally(new DoubleMatrix(regret), new DoubleMatrix(mqDiffs)), new DoubleMatrix(vminDiffs));

		String fileName = "vmin" + (int)vminvarInput + "mq" + (int)mqInput + "kwh" + (int)kwhPerUnit + "tstart" + (int)tstartTrue + "tcrit" + (int)tcritTrue + "qmax" +(int)qmaxInput + "qmin" + (int)(qminInput*100) + "bivar" + bayesInfVar ;
		FileHandler.safeDailyReport(dailyReport, fileHandleOut  + fileName + "daily.csv");
		FileHandler.safeTimeStepReport(timeStepReport, fileHandleOut + fileName + "timestep.csv");
		FileHandler.safeStoppingReport(stoppingReport, fileHandleOut + fileName + "stopping.csv");
		System.out.println("Done  " + runId);

		//		FileHandler.matrixToCsv(printOut, fileHandleOut);

	}

	//	public GP predictPrices(){
	//		String fileHandlePrices = prices;//"/users/balz/documents/workspace/masterarbeit/data/prices2.csv";
	//
	//		SquaredExponential c1 = new SquaredExponential();
	//		Periodic c2 = new Periodic();
	//		Matern m = new Matern();
	//		Multiplicative mult = new Multiplicative(c1, c2);
	//		Additive a1 = new Additive(mult, m);
	//		
	//		int noData = 96;
	//		double stepsize = 1./96.;
	//		double[] dataX = new double[noData];
	//		double[] dataY =  new double[noData];
	//		for(int i=0; i< dataX.length; i++){
	//			dataX[i] = i*stepsize;
	//		}
	//		for(int i=0; i< noData; i++){
	//			dataY[i] = 10;
	//		}
	//	
	//		double[] dataP = {2,1.5,2,1,1.2,0.2,0.2};//{2,1.5,1,1.2,0.2,0.2}
	//		double[] dataTest = new double[noData];// = {11,12,13,14,15,16};
	//		for(int i = 0; i < dataTest.length/10; i++){
	//			dataTest[i] = i*stepsize + noData*stepsize;
	//		}
	//		double nl = 0.05;		
	//		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);
	//		DoubleMatrix predictedPrices = new DoubleMatrix();
	//		DoubleMatrix P = new DoubleMatrix(dataP);
	//		CovarianceFunction cf = a1;
	//		int runs = 1;
	//		double cumulativeU = 0;
	//		double currentLoad = 0;
	//		int steps = 96;
	//		int trainSetSize = 1;
	//		int initialOffset = 0;
	//		double[] loads = new double[runs*steps];
	//		int[] actions = new int[runs*steps];
	//
	//		DoubleMatrix priceSimple1 = DoubleMatrix.ones(24).mul(40);
	//		DoubleMatrix priceSimple2 = DoubleMatrix.ones(24).mul(40);
	//		DoubleMatrix priceSimple3 = DoubleMatrix.ones(24).mul(0);
	//		DoubleMatrix priceSimple4 = DoubleMatrix.ones(24).mul(0);
	//		DoubleMatrix priceSimple = DoubleMatrix.concatVertically(priceSimple1, priceSimple2);
	//		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple3);
	//		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple4);
	//
	//		for(int i = 0; i < runs; i++){
	//			double[] xTrain = new double[steps*trainSetSize];
	//			for(int o = 0; o < steps*trainSetSize; o++){
	//				xTrain[o] = o*stepsize+initialOffset*stepsize;
	//			}
	//			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
	//			double[] xTest = new double[steps];
	//			for(int o = 0; o < steps; o++){
	//				xTest[o] = o*stepsize + (initialOffset+steps)*stepsize*trainSetSize;
	//			}
	//			DoubleMatrix xTestM = new DoubleMatrix(xTest);
	//
	//			
	//		
	//			DoubleMatrix yTrainMPrices = subVector(initialOffset, initialOffset+steps*trainSetSize, priceSamples);
	////			printMatrix(xTrainM);
	////			printMatrix(xTestM);
	//			GP priceGP = new GP(xTrainM,xTestM,P,cf,nl);
	//			priceGP.setup(yTrainMPrices);
	//			DoubleMatrix predMeanPrices = priceGP.getPredMean().add(20);
	//			DoubleMatrix predVarPrices = priceGP.getPredVar();
	//			EVMDP mdp = new EVMDP(predMeanPrices,predVarPrices, .5,steps);
	//
	//			mdp.work();
	//			// heat according to policy and update cumulative utility
	//			int tmp = initialOffset+steps*trainSetSize;
	//			for(int o = tmp; o < tmp + steps; o++){
	////				System.out.println(o  + " " + testmdp.priceToState(predMeanPrices.get(o-tmp))+ " " +  currentLoad );
	//				int action = mdp.getOptPolicy()[o-tmp][mdp.priceToState(predMeanPrices.get(o-tmp))][mdp.loadToState(currentLoad)];
	//				cumulativeU += mdp.rewards(currentLoad, action, priceSamples.get(o)+20,o);
	//				currentLoad = mdp.updateLoad(currentLoad, action);
	//
	//				loads[o-steps*trainSetSize-initialOffset] = currentLoad;
	//				actions[o-steps*trainSetSize-initialOffset] = action;
	//			}
	//			initialOffset += steps;
	//			if(i > 0){
	//				predictedPrices = DoubleMatrix.concatVertically(predictedPrices, predMeanPrices);
	//			} else {
	//				predictedPrices = predMeanPrices;
	//			}
	//
	//		}
	//		printMatrix(subVector(steps*trainSetSize, steps*trainSetSize+runs*steps, priceSamples).add(20));
	//		System.out.println();
	//
	//		printMatrix(predictedPrices);
	//		System.out.println();
	//
	//		System.out.print("[");
	//		for(int i = 0; i< loads.length; i++){
	//			System.out.print(loads[i] + "; ");
	//
	//		}
	//		System.out.println("]");
	//		System.out.print("[");
	//		for(int i = 0; i< actions.length; i++){
	//			System.out.print(actions[i] + "; ");
	//
	//		}
	//		System.out.println("]");
	//		System.out.println(cumulativeU);
	//		FileHandler.matrixToCsv(predictedPrices, out);
	//	}

	public double sampleFromNormal(double mean, double sd){
		NormalDistribution nd = new NormalDistribution(mean, sd);
		return nd.sample();
	}

	public boolean isInRange(double value, double goal, double range){
		if(Math.abs(value-goal)<= range){
			return true;
		}
		else{
			return false;
		}
	}

	public class SimulationThread implements Runnable{
		public EVMDP mdp;
		public int counter;
		public double value;
		public double expectedUtil;
		public int[] indexes;
		public Thread thread;
		private final Object lock = new Object();

		public SimulationThread(EVMDP emdp, int counter, double value, double expectedUtil, int[] indexes){
			this.mdp = emdp;
			this.counter = counter;
			this.value = value;
			this.expectedUtil = expectedUtil;
			this.indexes = indexes;
		}
		@Override
		public void run() {
			// TODO Auto-generated method stub
			mdp.fastSetup();
			double expectedSampleUtility = mdp.getqValues()[0][indexes[0]][indexes[1]][0];
			value = expectedSampleUtility - expectedUtil;
			//			System.out.println("Donezo");


		}
		public void start(){
			this.thread = new Thread(this);
			this.thread.start();
		}


	}


}
