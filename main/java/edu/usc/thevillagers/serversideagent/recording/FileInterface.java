package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;

public abstract class FileInterface {

	protected final File file;

	public FileInterface(File file) {
		this.file = file;
	}
	
	public abstract void write() throws IOException;
	public abstract void read() throws IOException;
	public abstract void clearData();
	public abstract boolean hasData();
}
