package br.ufrn.imd.selftraining.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import br.ufrn.imd.selftraining.results.InstanceResultStandard;
import br.ufrn.imd.selftraining.utils.Mathematics;
import weka.core.DenseInstance;
import weka.core.Instance;

public class SelfTrainingDws extends SelfTraining{

	public SelfTrainingDws(Dataset testSet, Dataset validationSet) {
		super(testSet, validationSet);
	}

	/**
	 * Join the p percent best instances (unlabeledSetJoinRate) according to
	 * confidence at each iteration but with weight from distance
	 * 
	 * @throws Exception
	 */
	public void runStandardDws() throws Exception {
		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;

		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();

			trainMainCLassifierOverLabeledSet();
			classifyInstancesStandardDws(this.unlabeledSet);

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
	 * Join the p percent best instances (unlabeledSetJoinRate) according to
	 * confidence at each iteration but with weight from distance.
	 * 
	 * The diff from runStandardDws() is the way the selection is performed. This "new" version considers all confidences and calcs the
	 * dws value to all classes to chose the best dws value for each instance. The old version only looks to the best confidence.
	 * 
	 * @throws Exception
	 */
	public void runStandardDwsNew() throws Exception {
		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;

		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();

			trainMainCLassifierOverLabeledSet();
			classifyInstancesStandardDwsNew(this.unlabeledSet);

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
	
	protected void classifyInstancesStandardDws(Dataset dataset) throws Exception {

		ArrayList<InstanceResultStandard> standardResults = new ArrayList<InstanceResultStandard>();
		Instance[] centroids = Mathematics.centroidsOf(this.labeledSet.getInstances());

		int amount = this.amountToJoin;

		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");

		InstanceResultStandard instanceResultStandard;

		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while (iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResultStandard = new InstanceResultStandard(instance);
			instanceResultStandard.addConfidences(this.mainClassifier.distributionForInstance(instance));

			Double distance = Mathematics.euclidianDistance(instance,
					centroids[instanceResultStandard.getBestClassIndex()]);
			instanceResultStandard.setFactor(instanceResultStandard.getBestConfidence() * (1 / distance));

			standardResults.add(instanceResultStandard);
		}

		Collections.sort(standardResults, InstanceResultStandard.factorComparatorDesc);

		for (InstanceResultStandard irs : standardResults) {
			sb.append(irs.outputDataToCsvWithDistanceFactor() + "\n");
		}

		if (this.unlabeledSet.getInstances().size() < amount * 2) {
			amount = this.unlabeledSet.getInstances().size();
		}

		for (int i = 0; i < amount; i++) {
			DenseInstance d = (DenseInstance) standardResults.get(i).getInstance().copy();
			d.setClassValue(standardResults.get(i).getBestClass());
			tempSet.addInstance(d); // CAUTION
			unlabeledSet.getInstances().remove(standardResults.get(i).getInstance());
		}

		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	protected void classifyInstancesStandardDwsNew(Dataset dataset) throws Exception {

		ArrayList<InstanceResultStandard> standardResults = new ArrayList<InstanceResultStandard>();
		Instance[] centroids = Mathematics.centroidsOf(this.labeledSet.getInstances());

		int amount = this.amountToJoin;

		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");

		InstanceResultStandard instanceResultStandard;

		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while (iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResultStandard = new InstanceResultStandard(instance);
			instanceResultStandard.addConfidences(this.mainClassifier.distributionForInstance(instance));

			ArrayList<Double> dwsValues = new ArrayList<Double>();
			
			for(int i = 0; i < centroids.length; i++) {
				Double distance = Mathematics.euclidianDistance(instance,
						centroids[i]);
				dwsValues.add(instanceResultStandard.getConfidences().get(i) * (1 / distance));
			}
			instanceResultStandard.addDwsValues(dwsValues);

			standardResults.add(instanceResultStandard);
		}

		Collections.sort(standardResults, InstanceResultStandard.bestDwsValueComparatorDesc);

		for (InstanceResultStandard irs : standardResults) {
			sb.append(irs.outputDataToCsvWithDwsValues() + "\n");
		}

		if (this.unlabeledSet.getInstances().size() < amount * 2) {
			amount = this.unlabeledSet.getInstances().size();
		}

		for (int i = 0; i < amount; i++) {
			DenseInstance d = (DenseInstance) standardResults.get(i).getInstance().copy();
			d.setClassValue(standardResults.get(i).getBestClass());
			tempSet.addInstance(d); // CAUTION
			unlabeledSet.getInstances().remove(standardResults.get(i).getInstance());
		}

		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
}
