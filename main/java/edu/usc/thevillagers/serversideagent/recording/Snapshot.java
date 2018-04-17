package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;

import net.minecraft.block.Block;
import net.minecraft.block.state.IBlockState;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.IBlockAccess;

public class Snapshot extends NBTFileInterface<IBlockState[]> {
	
	public Snapshot(File file) {
		super(file);
	}
	
	private int encodeBlockState(IBlockState state) {
		return Block.getStateId(state);
	}
	
	private IBlockState decodeBlockState(int id) {
		return Block.getStateById(id);
	}
	
	public void setDataFromWorld(IBlockAccess world, BlockPos from, BlockPos to) {
		BlockPos diff = to.subtract(from).add(1, 1, 1);
		data = new IBlockState[diff.getX() * diff.getY() * diff.getZ()];
		int index = 0;
		for(BlockPos p : BlockPos.getAllInBoxMutable(from, to)) {
			data[index++] = world.getBlockState(p);
		}
	}
	
	public void applyDataToWorld(ReplayWorldAccess world, BlockPos from, BlockPos to) {
		int index = 0;
		for(BlockPos p : BlockPos.getAllInBoxMutable(from, to)) {
			world.setBlockState(p, data[index++]);
		}
	}

	@Override
	protected void writeNBT(NBTTagCompound compound) throws IOException {
		int[] encodedData = new int[data.length];
		for(int i = 0; i < data.length; i ++)
			encodedData[i] = encodeBlockState(data[i]);
		compound.setIntArray("BlockStates", encodedData);
	}

	@Override
	protected void readNBT(NBTTagCompound compound) throws IOException {
		int[] encodedData = compound.getIntArray("BlockStates");
		if(encodedData == null) throw new IOException("Wrong file format");
		data = new IBlockState[encodedData.length];
		for(int i = 0; i < data.length; i++)
			data[i] = decodeBlockState(encodedData[i]);
	}
}
