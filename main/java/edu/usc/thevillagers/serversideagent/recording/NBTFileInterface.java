package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
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
		NBTTagCompound compound = new NBTTagCompound();
		writeNBT(compound);
		CompressedStreamTools.write(compound, file);
	}

	@Override
	public final void read() throws IOException {
		readNBT(CompressedStreamTools.read(file));
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
