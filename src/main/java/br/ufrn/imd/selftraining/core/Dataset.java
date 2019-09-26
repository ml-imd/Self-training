package br.ufrn.imd.selftraining.core;

import java.util.ArrayList;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Dataset {

	private String datasetName;
	private Instances instances;

	public Dataset() {
		
	}
	
	public Dataset(String pathAndDataSetName) throws Exception {
		this.instances = DataSource.read(pathAndDataSetName);
		this.datasetName = instances.relationName();
		instances.setClassIndex(instances.numAttributes() - 1);
	}

	public Dataset(String pathAndDataSetName, int classIndex) throws Exception {
		DataSource data = new DataSource(pathAndDataSetName);
		instances = data.getDataSet();
		this.datasetName = instances.relationName();
		instances.setClassIndex(classIndex);
	}

	public void shuffleInstances(int seed) {
		this.instances.randomize(new Random(seed));
	}

	public Dataset(Dataset dataset) {
		this.instances = new Instances(dataset.getInstances());
		this.datasetName = instances.relationName();
	}

	public Dataset(Instances instances) {
		this.instances = new Instances(instances);
		this.datasetName = instances.relationName();
	}

	public void addInstance(Instance instance) {
		Instance a = instance;
		this.instances.add(a);
	}

	public void addInstance(Instance instance, double prediction) {
		Instance a = instance;
		a.setClassValue(prediction);
		this.instances.add(a);
	}

	public void clearInstances() {
		this.instances.clear();
	}

	public String getDatasetName() {
		return datasetName;
	}

	public void setDatasetName(String datasetName) {
		this.datasetName = datasetName;
	}

	public Instances getInstances() {
		return instances;
	}

	public void setInstances(Instances instances) {
		this.instances = instances;
	}

	public static ArrayList<Dataset> splitDataset(Dataset dataset, int numberOfParts) {

		ArrayList<Dataset> splitedDataset = new ArrayList<Dataset>();
		ArrayList<Instance> myData = new ArrayList<Instance>();
		ArrayList<Instance> part = new ArrayList<Instance>();

		int size = dataset.getInstances().size() / numberOfParts;

		for (Instance i : dataset.getInstances()) {
			myData.add(i);
		}
		int i = 0;
		int control = 0;
		for (i = 0; i < myData.size(); i++) {
			part.add(myData.get(i));
			if (part.size() == size) {
				Dataset d = new Dataset();
				d.setDatasetName(dataset.getInstances().relationName());
				d.setInstances(new Instances(dataset.getInstances()));
				d.getInstances().clear();
				d.getInstances().addAll(part);

				splitedDataset.add(d);
				part = new ArrayList<Instance>();
				control = i;
			}
		}

		int x = 0;

		while (control < (myData.size() - 1)) {
			splitedDataset.get(x).getInstances().add(myData.get(control));
			control++;
			x++;
			if (x == (splitedDataset.size() - 1)) {
				x = 0;
			}
		}

		return splitedDataset;
	}

	public static Dataset joinDatasets(ArrayList<Dataset> folds) {
		Dataset dataset = new Dataset(folds.get(0));
		for (int i = 1; i < folds.size(); i++) {
			dataset.getInstances().addAll(folds.get(i).getInstances());
		}
		return dataset;
	}

}
