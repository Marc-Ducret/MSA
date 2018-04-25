package edu.usc.thevillagers.serversideagent.proxy;

import com.mojang.authlib.GameProfile;

import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.world.World;

public interface Proxy {
	
	public EntityPlayer createReplayEntityPlayer(World world, GameProfile profile);
	
}
