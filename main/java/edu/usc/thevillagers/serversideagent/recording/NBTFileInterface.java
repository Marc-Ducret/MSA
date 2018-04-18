package edu.usc.thevillagers.serversideagent.recording;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import net.minecraft.nbt.CompressedStreamTools;
import net.minecraft.nbt.NBTTagCompound;

public abstract class NBTFileInterface<T> extends FileInterface {
	
	public T data;

	public NBTFileInterface(File file) {
		super(file);
	}
	
	protected abstract void writeNBT(NBTTagCompound compound) throws IOException;
	protected abstract void readNBT(NBTTagCompound compound) throws IOException;

	@Override
	public final void write() throws IOException {
		
		DataOutputStream stream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
		try {
			NBTTagCompound compound = new NBTTagCompound();
			writeNBT(compound);
			CompressedStreamTools.write(compound, stream);
		} finally {
			stream.close();
		}
		
		
		
	}

	@Override
	public final void read() throws IOException {
		DataInputStream stream = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
		try {
			NBTTagCompound comp = CompressedStreamTools.read(stream);
			readNBT(comp);
		} finally {
			stream.close();
		}
	}

	@Override
	public final void clearData() {
		data = null;
	}

	@Override
	public final boolean hasData() {
		return data != null;
	}
}
