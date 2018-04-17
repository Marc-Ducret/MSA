package edu.usc.thevillagers.serversideagent.recording;

import net.minecraft.block.state.IBlockState;
import net.minecraft.init.Blocks;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.EnumFacing;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.IBlockAccess;
import net.minecraft.world.WorldType;
import net.minecraft.world.biome.Biome;

public class ReplayWorldAccess implements IBlockAccess {
	
	public final BlockPos from, to, diff;
	private final IBlockState[] blockBuffer;
	
	public ReplayWorldAccess(BlockPos from, BlockPos to) {
		this.from = from.toImmutable();
		this.to = to.toImmutable();
		
		diff = to.subtract(from).add(1, 1, 1);
		blockBuffer = new IBlockState[diff.getX() * diff.getY() * diff.getZ()];
	}

	@Override
	public TileEntity getTileEntity(BlockPos pos) {
		return null;
	}

	@Override
	public int getCombinedLight(BlockPos pos, int lightValue) {
		lightValue = Math.max(lightValue, 0x7);
		if(isAirBlock(pos)) lightValue = Math.max(lightValue, 0xA);
		return lightValue << 4;
	}
	
	private int index(BlockPos pos) {
		BlockPos p = pos.subtract(from);
		if(		p.getX() < 0 || p.getX() >= diff.getX() ||
				p.getY() < 0 || p.getY() >= diff.getY() ||
				p.getZ() < 0 || p.getZ() >= diff.getZ())
			return -1;
		return p.getX() + diff.getX() * (p.getY() + diff.getY() * p.getZ());
	}
	
	@Override
	public IBlockState getBlockState(BlockPos pos) {
		int index = index(pos);
		if(index < 0) return Blocks.AIR.getDefaultState();
		return blockBuffer[index];
	}
	
	public void setBlockState(BlockPos pos, IBlockState state) {
		int index = index(pos);
		if(index >= 0)
			blockBuffer[index] = state;
	}

	@Override
	public boolean isAirBlock(BlockPos pos) {
		return getBlockState(pos).getBlock() == Blocks.AIR;
	}

	@Override
	public Biome getBiome(BlockPos pos) {
		return Biome.getBiome(1);
	}

	@Override
	public int getStrongPower(BlockPos pos, EnumFacing direction) {
		return 0;
	}

	@Override
	public WorldType getWorldType() {
		return WorldType.CUSTOMIZED;
	}

	@Override
	public boolean isSideSolid(BlockPos pos, EnumFacing side, boolean _default) {
		int index = index(pos);
		if(index < 0) return _default;
		return blockBuffer[index].isSideSolid(this, pos, side);
	}
}
