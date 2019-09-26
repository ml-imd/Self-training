package br.ufrn.imd.selftraining.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import br.ufrn.imd.selftraining.results.FoldResult;
import br.ufrn.imd.selftraining.results.InstanceResult;
import br.ufrn.imd.selftraining.results.InstanceResultStandard;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class SelfTrainingMachine{

	private Dataset validationSet;
	private Dataset testSet;

	private Dataset tempSet;

	private int labeledSetPercentual = 10;
	private Dataset labeledSet;
	private Dataset unlabeledSet;
	
	private int unlabeledSetJoinRate = 10;
	private int amountToJoin = 0;
	
	private ArrayList<Classifier> pool;
	private Classifier mainClassifier;

	private int goodClassifiedInstances = 0;
	
	private double agreementThreshold = 75; //percentual

	private FoldResult result;
	private String history;
	private String iterationInfo;
	
	public SelfTrainingMachine(Dataset testSet, Dataset validationSet) {
		
		this.result = new FoldResult();
		this.history = new String();
		
		this.pool = new ArrayList<Classifier>();
		populatePool();

		this.validationSet = new Dataset(validationSet);
		this.testSet = new Dataset(testSet);
		this.tempSet = new Dataset(testSet);
		this.tempSet.getInstances().clear();

		splitDatasetStratified();
		createMainClassifier();
	}

	public void runVersionOne() throws Exception {
		
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();
			
			trainMainCLassifierOverLabeledSet();
			trainClassifiersPool();
			classifyInstancesAndCheckAgreement(this.unlabeledSet);

			if (tempSet.getInstances().size() == 0) {
				break;
			}
			
			classifyBestWithMainClassifier();
			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances);
			clearTempSet();
			i++;
			printIterationInfo();
		}
		mainClassifierJob();
	}

	public void runVersionTwo() throws Exception {
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			trainMainCLassifierOverLabeledSet();
			trainClassifiersPool();
			classifyInstancesCheckAgreementPool(this.unlabeledSet);

			if (tempSet.getInstances().size() == 0) {
				break;
			}
			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances);
			clearTempSet();
			i++;
			printIterationInfo();
			addIterationInfoToHistory();
		}
		mainClassifierJob();
	}
	
	//Join only the n best instances according to confidence
	public void runStandard() throws Exception {
		
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
			result.addIterationInfo(this.goodClassifiedInstances);
			
			clearTempSet();
			i++;
			printIterationInfo();
		}
		mainClassifierJob();
	}
	
	//Join the p percent best instances according to confidence at each iteration
	public void runStandard2() throws Exception {
		
		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;
		
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();
			
			trainMainCLassifierOverLabeledSet();
			classifyInstancesStandard2(this.unlabeledSet);

			if (tempSet.getInstances().size() == 0) {
				break;
			}
			
			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances);
			
			clearTempSet();
			i++;
			printIterationInfo();
		}
		mainClassifierJob();
	}
	
	
	private void trainMainCLassifierOverLabeledSet() throws Exception {
		this.mainClassifier.buildClassifier(this.labeledSet.getInstances());
	}

	public void trainClassifiersPool() throws Exception {

		for (Classifier c: pool) {
			c.buildClassifier(this.labeledSet.getInstances());
		}

	}

	private void classifyInstancesAndCheckAgreement(Dataset dataset) throws Exception {
		
		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		double agreementValue = pool.size() * agreementThreshold / 100;
		int value = (int) agreementValue;
		
		InstanceResult instanceResult;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResult = new InstanceResult(instance);
			for(Classifier c: this.pool) {
				instanceResult.addPrediction(c.classifyInstance(instance));
			}
			
			if(instanceResult.getBestAgreement() >= value) {
				tempSet.addInstance((Instance)instance.copy()); //CAUTION
				iterator.remove();
			}
			sb.append(instanceResult.outputDataToCsv() + "\n");
		}
		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	private void classifyInstancesCheckAgreementPool(Dataset dataset) throws Exception {
		
		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		double agreementValue = pool.size() * agreementThreshold / 100;
		int value = (int) agreementValue;
		
		InstanceResult instanceResult;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResult = new InstanceResult(instance);
			for(Classifier c: this.pool) {
				instanceResult.addPrediction(c.classifyInstance(instance));
			}
			
			if(instanceResult.getBestAgreement() >= value) {
				DenseInstance d = (DenseInstance) instance.copy();
				d.setClassValue(instanceResult.getBestClass());
				
				tempSet.addInstance(d); //CAUTION
				iterator.remove();
			}
			sb.append(instanceResult.outputDataToCsv() + "\n");
		}
		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	private void classifyInstancesStandard(Dataset dataset) throws Exception {
		
		ArrayList<InstanceResultStandard> standardResults = new ArrayList<InstanceResultStandard>();
		Double bestConfidence = 0.0;
		
		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		InstanceResultStandard instanceResultStandard;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResultStandard = new InstanceResultStandard(instance);
			instanceResultStandard.addConfidences(this.mainClassifier.distributionForInstance(instance));
			standardResults.add(instanceResultStandard);
			
			if(instanceResultStandard.getBestConfidence() > bestConfidence) {
				bestConfidence = instanceResultStandard.getBestConfidence();
			}

			sb.append(instanceResultStandard.outputDataToCsv() + "\n");
		}
		
		for(InstanceResultStandard irs: standardResults) {
			if(irs.getBestConfidence() >= bestConfidence) {
				DenseInstance d = (DenseInstance) irs.getInstance().copy();
				d.setClassValue(irs.getBestClass());
				tempSet.addInstance(d); //CAUTION
				unlabeledSet.getInstances().remove(irs.getInstance());
			}
		}
		
		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	private void classifyInstancesStandard2(Dataset dataset) throws Exception {
		
		ArrayList<InstanceResultStandard> standardResults = new ArrayList<InstanceResultStandard>();
		int amount = this.amountToJoin;

		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		InstanceResultStandard instanceResultStandard;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResultStandard = new InstanceResultStandard(instance);
			instanceResultStandard.addConfidences(this.mainClassifier.distributionForInstance(instance));
			standardResults.add(instanceResultStandard);

			sb.append(instanceResultStandard.outputDataToCsv() + "\n");
		}
		
		Collections.sort(standardResults, InstanceResultStandard.bestConfidenceComparatorDesc);
		
		if(this.unlabeledSet.getInstances().size() < amount*2) {
			amount = this.unlabeledSet.getInstances().size();
		}
		
		for(int i = 0; i < amount; i++) {
			DenseInstance d = (DenseInstance) standardResults.get(i).getInstance().copy();
			d.setClassValue(standardResults.get(i).getBestClass());
			tempSet.addInstance(d); //CAUTION
			unlabeledSet.getInstances().remove(standardResults.get(i).getInstance());
		}

		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	private void clearTempSet() {
		this.tempSet.getInstances().clear();
	}

	private void classifyBestWithMainClassifier() throws Exception {
		for (Instance i : this.tempSet.getInstances()) {
			double x = this.mainClassifier.classifyInstance(i);
			i.setClassValue(x);
		}
	}

	private void joinClassifiedWithLabeledSet() {
		labeledSet.getInstances().addAll(this.tempSet.getInstances());
	}

	public void createMainClassifier() {
		// weka.classifiers.trees.J48 -C 0.05 -M 2 (74.4792)
		J48 j48 = new J48();
		try {
			j48.setOptions(weka.core.Utils.splitOptions("-C 0.05 -M 2"));
		} catch (Exception e) {
			e.printStackTrace();
		}
		this.mainClassifier = (J48) j48;
	}

	public void populatePool() {
		J48 j48a = new J48();
		J48 j48b = new J48();
		J48 j48c = new J48();
		J48 j48d = new J48();

		NaiveBayes nb1 = new NaiveBayes();
		NaiveBayes nb2 = new NaiveBayes();
		NaiveBayes nb3 = new NaiveBayes();

		IBk ibk1 = new IBk();
		IBk ibk2 = new IBk();
		IBk ibk3 = new IBk();
		IBk ibk4 = new IBk();
		IBk ibk5 = new IBk();

		SMO smo1 = new SMO();
		SMO smo2 = new SMO();
		SMO smo3 = new SMO();
		SMO smo4 = new SMO();
		SMO smo5 = new SMO();

		DecisionTable dt1 = new DecisionTable();
		DecisionTable dt2 = new DecisionTable();
		DecisionTable dt3 = new DecisionTable();

		try {

			j48a.setOptions(weka.core.Utils.splitOptions("-C 0.05 -M 2"));
			j48b.setOptions(weka.core.Utils.splitOptions("-C 0.10 -M 2"));
			j48c.setOptions(weka.core.Utils.splitOptions("-C 0.20 -M 2"));
			j48d.setOptions(weka.core.Utils.splitOptions("-C 0.25 -M 2"));

			nb1.setOptions(weka.core.Utils.splitOptions(""));
			nb2.setOptions(weka.core.Utils.splitOptions("-K"));
			nb3.setOptions(weka.core.Utils.splitOptions("-D"));

			ibk1.setOptions(weka.core.Utils.splitOptions(
					"-K 1 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""));
			ibk2.setOptions(weka.core.Utils.splitOptions(
					"-K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""));
			ibk3.setOptions(weka.core.Utils.splitOptions(
					"-K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.ManhattanDistance -R first-last\\\"\""));
			ibk4.setOptions(weka.core.Utils.splitOptions(
					"-K 5 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""));
			ibk5.setOptions(weka.core.Utils.splitOptions(
					"-K 5 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.ManhattanDistance -R first-last\\\"\""));

			smo1.setOptions(weka.core.Utils.splitOptions(
					"-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));
			smo2.setOptions(weka.core.Utils.splitOptions(
					"-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.NormalizedPolyKernel -E 2.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));
			smo3.setOptions(weka.core.Utils.splitOptions(
					"-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));
			smo4.setOptions(weka.core.Utils.splitOptions(
					"-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.Puk -O 1.0 -S 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));
			smo5.setOptions(weka.core.Utils.splitOptions(
					"-C 0.8 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));

			dt1.setOptions(weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 5\""));
			dt2.setOptions(weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 3\""));
			dt3.setOptions(weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 7\""));

		} catch (Exception e) {
			e.printStackTrace();
		}

		this.pool.add(j48a);
		this.pool.add(j48b);
		this.pool.add(j48c);
		this.pool.add(j48d);

		this.pool.add(nb1);
		this.pool.add(nb2);
		this.pool.add(nb3);

		this.pool.add(ibk1);
		this.pool.add(ibk2);
		this.pool.add(ibk3);
		this.pool.add(ibk4);
		this.pool.add(ibk5);

		this.pool.add(smo1);
		this.pool.add(smo2);
		this.pool.add(smo3);
		this.pool.add(smo4);
		this.pool.add(smo5);

		this.pool.add(dt1);
		this.pool.add(dt2);
		this.pool.add(dt3);
	}

	public void splitDatasetStratified() {
		testSet.getInstances().stratify(10);
		this.labeledSet = new Dataset(testSet.getInstances().testCV(10, 0));
		this.unlabeledSet = new Dataset(testSet.getInstances().trainCV(10, 0));
	}

	public void splitByPercentage() {
		int total = testSet.getInstances().size() * (this.labeledSetPercentual / 100);

		this.labeledSet = new Dataset(new Instances(this.testSet.getInstances(), 0, total));

		this.unlabeledSet = new Dataset(new Instances(this.testSet.getInstances(), 0, 1));
		this.unlabeledSet.getInstances().clear();

		for (int i = total; i < this.testSet.getInstances().size(); i++) {
			this.unlabeledSet.getInstances().add(this.testSet.getInstances().get(i));
		}
	}
	
	private void mainClassifierJob() throws Exception{
		//this.mainClassifier.buildClassifier(labeledSet.getInstances());
		Measures measures = new Measures(this.mainClassifier, this.labeledSet.getInstances(), this.validationSet.getInstances());
		
		result.setAccuracy(measures.getAccuracy());
		result.setError(measures.getError());
		result.setfMeasure(measures.getFmeasureMean());
		result.setPrecision(measures.getPrecisionMean()); 
		result.setRecall(measures.getRecallMean()); 	
	}
	
	private void printIterationInfo() {
		System.out.println(this.iterationInfo);
	}
	
	private void addIterationInfoToHistory() {
		addToHistory(this.iterationInfo);
	}
	
	private void addToHistory(String string) {
		StringBuilder sb = new StringBuilder();
		sb.append(this.history);
		sb.append(string);
		
		this.history = new String(sb.toString());
	}
	
	private void generateIterationInfo(int iteration) {
		
		StringBuilder sb = new StringBuilder();
		
		sb.append("--------------------------------------------------\n");
		sb.append("@ SELFTRAINING ITERATION: " + iteration + "\n");
		sb.append("--------------------------------------------------\n");
		sb.append("\n");
		sb.append("@ LABELED: " + this.labeledSet.getInstances().size() + "\n");
		sb.append("@ UNLABELED: " + this.unlabeledSet.getInstances().size() + "\n");
		if(iteration > 1) {
			sb.append("@ PASSED OVER AGREEMENT TRESHOLD (last iteration): " + goodClassifiedInstances + "\n");
		}
		sb.append("--------------------------------------------------\n");
		this.iterationInfo = new String(sb.toString());
	}
	
	// GETs and SETs
 	public Dataset getValidationSet() {
		return validationSet;
	}

	public void setValidationSet(Dataset validationSet) {
		this.validationSet = validationSet;
	}

	public Dataset getTestSet() {
		return testSet;
	}

	public void setTestSet(Dataset testSet) {
		this.testSet = testSet;
	}

	public int getLabeledSetPercentual() {
		return labeledSetPercentual;
	}

	public void setLabeledSetPercentual(int labeledSetPercentual) {
		this.labeledSetPercentual = labeledSetPercentual;
	}

	public Dataset getLabeledSet() {
		return labeledSet;
	}

	public void setLabeledSet(Dataset labeledSet) {
		this.labeledSet = labeledSet;
	}

	public Dataset getUnlabeledSet() {
		return unlabeledSet;
	}

	public void setUnlabeledSet(Dataset unlabeledSet) {
		this.unlabeledSet = unlabeledSet;
	}

	public ArrayList<Classifier> getPool() {
		return pool;
	}

	public void setPool(ArrayList<Classifier> pool) {
		this.pool = pool;
	}

	public Classifier getMainClassifier() {
		return mainClassifier;
	}

	public void setMainClassifier(Classifier mainClassifier) {
		this.mainClassifier = mainClassifier;
	}

	public double getAgreementThreshold() {
		return agreementThreshold;
	}

	public void setAgreementThreshold(double agreementThreshold) {
		this.agreementThreshold = agreementThreshold;
	}

	public FoldResult getResult() {
		return result;
	}

	public void setResult(FoldResult result) {
		this.result = result;
	}

	public String getHistory() {
		return history;
	}

	public void setHistory(String history) {
		this.history = history;
	}
	
}
