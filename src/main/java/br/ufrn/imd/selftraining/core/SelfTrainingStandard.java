package br.ufrn.imd.selftraining.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;

import br.ufrn.imd.selftraining.results.InstanceResultStandard;
import weka.core.DenseInstance;
import weka.core.Instance;

public class SelfTrainingStandard extends SelfTraining {

	public SelfTrainingStandard(Dataset testSet, Dataset validationSet) {
		super(testSet, validationSet);
	}

	/**
	 * Join the p percent best instances (unlabeledSetJoinRate) according to
	 * confidence at each iteration
	 * 
	 * @throws Exception
	 */
	public void runStandard() throws Exception {

		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;

		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();

			trainMainCLassifierOverLabeledSet();
			classifyInstancesStandard(this.unlabeledSet);

			if (tempSet.getInstances().size() == 0) {
				break;
			}

			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances, this.missClassifiedInstances);

			clearTempSet();
			i++;
			printIterationInfo();
		}
		mainClassifierJob();
	}

	/**
	 * Join only the n best instances that get the best confidence of running
	 * iteration
	 * 
	 * @throws Exception
	 */
	public void runStandardLazy() throws Exception {

		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();

			trainMainCLassifierOverLabeledSet();
			classifyInstancesStandardLazy(this.unlabeledSet);

			if (tempSet.getInstances().size() == 0) {
				break;
			}

			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances, this.missClassifiedInstances);

			clearTempSet();
			i++;
			printIterationInfo();
		}
		mainClassifierJob();
	}
	
	/**
	 * Join the p percent (unlabeledSetJoinRate) according to a random selection
	 * 
	 * @throws Exception
	 */
	public void runRandom() throws Exception {

		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;

		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();

			trainMainCLassifierOverLabeledSet();
			
			//select instances randomly
			selectInstancesByRandom();
			
			if (tempSet.getInstances().size() == 0) {
				break;
			}
			
			//classify selected instances with the main classsifier
			classifyInstancesFromTempWithMainClassifier();
			
			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances, this.missClassifiedInstances);

			this.goodClassifiedInstances = 0;
			this.missClassifiedInstances = 0;
			
			clearTempSet();
			i++;
			//printIterationInfo();
		}
		mainClassifierJob();
	}

	protected void classifyInstancesStandard(Dataset dataset) throws Exception {

		this.missClassifiedInstances = 0;

		ArrayList<InstanceResultStandard> standardResults = new ArrayList<InstanceResultStandard>();
		int amount = this.amountToJoin;

		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");

		InstanceResultStandard instanceResultStandard;

		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while (iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResultStandard = new InstanceResultStandard(instance);
			instanceResultStandard.addConfidences(this.mainClassifier.distributionForInstance(instance));
			standardResults.add(instanceResultStandard);

			sb.append(instanceResultStandard.outputDataToCsv() + "\n");
		}

		Collections.sort(standardResults, InstanceResultStandard.bestConfidenceComparatorDesc);

		if (this.unlabeledSet.getInstances().size() < amount * 2) {
			amount = this.unlabeledSet.getInstances().size();
		}

		for (int i = 0; i < amount; i++) {
			DenseInstance d = (DenseInstance) standardResults.get(i).getInstance().copy();
			d.setClassValue(standardResults.get(i).getBestClass());
			tempSet.addInstance(d); // CAUTION

			if (standardResults.get(i).getInstance().classValue() != standardResults.get(i).getBestClass()) {
				this.missClassifiedInstances += 1;
			}

			unlabeledSet.getInstances().remove(standardResults.get(i).getInstance());
		}

		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}

	protected void classifyInstancesStandardLazy(Dataset dataset) throws Exception {

		ArrayList<InstanceResultStandard> standardResults = new ArrayList<InstanceResultStandard>();
		Double bestConfidence = 0.0;

		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");

		InstanceResultStandard instanceResultStandard;

		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while (iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResultStandard = new InstanceResultStandard(instance);
			instanceResultStandard.addConfidences(this.mainClassifier.distributionForInstance(instance));
			standardResults.add(instanceResultStandard);

			if (instanceResultStandard.getBestConfidence() > bestConfidence) {
				bestConfidence = instanceResultStandard.getBestConfidence();
			}

			sb.append(instanceResultStandard.outputDataToCsv() + "\n");
		}

		for (InstanceResultStandard irs : standardResults) {
			if (irs.getBestConfidence() >= bestConfidence) {
				DenseInstance d = (DenseInstance) irs.getInstance().copy();
				d.setClassValue(irs.getBestClass());
				tempSet.addInstance(d); // CAUTION
				unlabeledSet.getInstances().remove(irs.getInstance());
			}
		}

		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}

	protected void selectInstancesByRandom() {
		this.missClassifiedInstances = 0;

		int amount = this.amountToJoin;
		int alreadySelected = 0;
		Random random = new Random();		
		
		if (this.unlabeledSet.getInstances().size() < amount * 2) {
			amount = this.unlabeledSet.getInstances().size();
		}
		
		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		//Select instances from unlabelled set
		while (alreadySelected < amount) {
			
			int size = unlabeledSet.getInstances().size();
			int drawed = random.nextInt(size);
			
			DenseInstance d = (DenseInstance) this.unlabeledSet.getInstances().get(drawed);
			tempSet.addInstance(d); // CAUTION
			unlabeledSet.getInstances().remove(drawed);
			
			alreadySelected++;
			
			sb.append(d.toString() + "\n");
		}
		
		this.goodClassifiedInstances = alreadySelected;
				
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	protected void classifyInstancesFromTempWithMainClassifier() throws Exception {
		
		for(Instance i: this.tempSet.getInstances()) {
			double d = this.mainClassifier.classifyInstance(i);
			
			if(d != i.classValue()) {
				this.missClassifiedInstances++;
			}
			
			i.setClassValue(d);
		}
	}
	
}
