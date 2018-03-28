package edu.usc.thevillagers.serversideagent;

import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.server.MinecraftServer;

public class CommandFastTick extends CommandBase {
	
	private final ServerSideAgentMod mod;
	
	public CommandFastTick(ServerSideAgentMod mod) {
		this.mod = mod;
	}

	@Override
	public String getName() {
		return "ft";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/ft <ticks>";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		int ticks = parseInt(args[0]);
		mod.requestFastTick(ticks, sender);
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
}