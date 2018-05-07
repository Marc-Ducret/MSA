package edu.usc.thevillagers.serversideagent.recording;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import net.minecraft.nbt.CompressedStreamTools;
import net.minecraft.nbt.NBTTagCompound;

/**
 * A specific type of {@link FileInterface} where the file format is NBT.
 * @param <T> The memory representation of the file's contents.
 */
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
		file.createNewFile();
		writeToFile(compound, file);
	}

	@Override
	public final void read() throws IOException {
		NBTTagCompound comp = readFromFile(file);
		readNBT(comp);
	}

	@Override
	public final void clearData() {
		data = null;
	}

	@Override
	public final boolean hasData() {
		return data != null;
	}
	
	public static void writeToFile(NBTTagCompound compound, File file) throws IOException {
		DataOutputStream stream = new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(file))));
		try {
			CompressedStreamTools.write(compound, stream);
		} finally {
			stream.close();
		}
	}
	
	public static NBTTagCompound readFromFile(File file) throws IOException {
		
		DataInputStream stream = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(file))));
		try {
			return CompressedStreamTools.read(stream);
		} finally {
			stream.close();
		}
	}
}
