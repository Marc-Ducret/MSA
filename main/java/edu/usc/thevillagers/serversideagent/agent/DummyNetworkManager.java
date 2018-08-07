package edu.usc.thevillagers.serversideagent.agent;

import io.netty.util.concurrent.Future;
import io.netty.util.concurrent.GenericFutureListener;
import net.minecraft.network.EnumPacketDirection;
import net.minecraft.network.NetworkManager;
import net.minecraft.network.Packet;

/**
 * EntityPlayer requires a NetworkManager, this one disable packets and is used by EntityAgent.
 */
public class DummyNetworkManager extends NetworkManager {

	public DummyNetworkManager() {
		super(EnumPacketDirection.SERVERBOUND);
	}
	
	@Override
	public void sendPacket(Packet<?> packetIn) {}
	
	@Override
	public void sendPacket(Packet<?> packetIn, GenericFutureListener<? extends Future<? super Void>> listener,
			@SuppressWarnings("unchecked") GenericFutureListener<? extends Future<? super Void>>... listeners) {}
	
}
