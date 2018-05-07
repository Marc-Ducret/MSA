package edu.usc.thevillagers.serversideagent.proxy;

import com.mojang.authlib.GameProfile;

import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.world.World;

/**
 * Allows for calls that depend on weather the client or dedicated server is running.
 */
public interface Proxy {
	
	public EntityPlayer createReplayEntityPlayer(World world, GameProfile profile);
	
}
