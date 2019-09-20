package br.ufrn.imd.selftraining.bc;

import br.ufrn.imd.selftraining.enums.ClassifierType;

public abstract class Classifier {

	protected String name;
	protected ClassifierType classifierType;
	protected String classifierId;

	public Classifier() {
		this.name = new String();
	}

	public abstract void setParametersByDefault();
	public abstract void buildClassifierId();

	public ClassifierType getClassifierType() {
		return classifierType;
	}

	public void setClassifierType(ClassifierType classifierType) {
		this.classifierType = classifierType;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public String getClassifierId() {
		return classifierId;
	}

	public void setClassifierId(String classifierId) {
		this.classifierId = classifierId;
	}

}
