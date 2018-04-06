package edu.usc.thevillagers.serversideagent.env;

import net.minecraft.block.BlockColored;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.EnumFacing;
import net.minecraft.util.math.BlockPos;

public class EnvironmentParkourRandom extends EnvironmentParkour {
	
	@Override
	public void reset() {
		super.reset();
		generate();
	}

	private void generate() {
		for(int z = 1; z <= length; z++) {
			for(int x =- width; x <= width; x++) {
				world.setBlockState(ref.add(x, -1, z), Blocks.AIR.getDefaultState());
			}
		}
		BlockPos pos = ref.add(0, -1, 1);
		int blocks = length + world.rand.nextInt(2 * width);
		EnumFacing[] dirs = new EnumFacing[] {EnumFacing.EAST, EnumFacing.WEST, EnumFacing.SOUTH};
		for(int i = 0; i < blocks; i ++) {
			world.setBlockState(pos, Blocks.CONCRETE.getDefaultState());
			EnumFacing dir = dirs[world.rand.nextInt(dirs.length)];
			BlockPos p = pos.offset(dir);
			if(p.getX() >= ref.getX() - width && p.getX() <= ref.getX() + width && p.getZ() > ref.getZ() && p.getZ() <= ref.getZ() + length)
				pos = p;
		}
		world.setBlockState(pos, Blocks.CONCRETE.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.LIME));
	}
}
